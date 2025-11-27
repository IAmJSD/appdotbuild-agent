"""Dagger-based app generation pipeline with caching and parallelism."""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import dagger

logger = logging.getLogger(__name__)


def _check_binary_format(binary_path: Path) -> None:
    """Check if binary is Linux-compatible for container execution.

    Raises:
        RuntimeError: If binary is not Linux ELF format
    """
    try:
        result = subprocess.run(
            ["file", str(binary_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout.lower()

        if "mach-o" in output or "darwin" in output:
            raise RuntimeError(
                f"Binary {binary_path} is macOS format (Mach-O), but Dagger runs Linux containers.\n"
                f"Please provide a Linux build. For Go: GOOS=linux GOARCH=arm64 go build ...\n"
                f"Output from 'file': {result.stdout.strip()}"
            )

        if "elf" not in output:
            logger.warning(
                f"Binary {binary_path} may not be Linux-compatible: {result.stdout.strip()}"
            )
    except FileNotFoundError:
        # 'file' command not available, skip check
        pass


class DaggerAppGenerator:
    """Runs app generation in Dagger container with caching."""

    def __init__(
        self,
        mcp_binary: Path,
        output_dir: Path,
        stream_logs: bool = True,
    ):
        _check_binary_format(mcp_binary)
        self.mcp_binary = mcp_binary
        self.output_dir = output_dir
        self.stream_logs = stream_logs

    async def generate_single(
        self,
        prompt: str,
        app_name: str,
        backend: str = "claude",
        model: str | None = None,
        mcp_args: list[str] | None = None,
    ) -> tuple[Path | None, Path]:
        """Generate single app, export app dir + logs.

        Returns:
            tuple of (app_dir or None, log_file) paths on host.
            app_dir is None if agent didn't create an app.
        """
        cfg = dagger.Config(log_output=sys.stderr) if self.stream_logs else dagger.Config()
        async with dagger.Connection(cfg) as client:
            container = await self._build_container(client)

            # path inside container for generated app
            app_output = f"/workspace/app/{app_name}"

            # build command using container_runner.py (already in image via Dockerfile COPY)
            cmd = [
                "python",
                "cli/generation/container_runner.py",
                prompt,
                f"--app_name={app_name}",
                f"--backend={backend}",
            ]
            if model:
                cmd.append(f"--model={model}")
            if mcp_args:
                cmd.append(f"--mcp_args={json.dumps(mcp_args)}")

            # ensure log directory exists and create app output dir
            container = container.with_exec(["mkdir", "-p", "/workspace/logs", "/workspace/app"])

            # run generation
            result = container.with_exec(cmd)

            # capture combined stdout/stderr as log
            log_content = await result.stdout()
            stderr_content = await result.stderr()
            full_log = f"{log_content}\n\n=== STDERR ===\n{stderr_content}" if stderr_content else log_content

            # save log to host
            log_file_local = self.output_dir / "logs" / f"{app_name}.log"
            log_file_local.parent.mkdir(parents=True, exist_ok=True)
            log_file_local.write_text(full_log)

            # export app directory (if it exists)
            app_dir_local = self.output_dir / app_name
            try:
                await result.directory(app_output).export(str(app_dir_local))
            except dagger.QueryError as e:
                if "no such file or directory" in str(e):
                    # agent didn't create an app directory (e.g. just answered a question)
                    return None, log_file_local
                raise

            return app_dir_local, log_file_local

    async def generate_bulk(
        self,
        prompts: dict[str, str],
        backend: str = "claude",
        model: str | None = None,
        mcp_args: list[str] | None = None,
        max_concurrency: int = 4,
    ) -> list[tuple[str, Path | None, Path | None, str | None]]:
        """Generate multiple apps with Dagger parallelism.

        Args:
            prompts: dict mapping app_name to prompt
            backend: "claude" or "litellm"
            model: model name (required for litellm)
            mcp_args: optional MCP server args
            max_concurrency: max parallel generations

        Returns:
            list of (app_name, app_dir, log_file, error) tuples
        """
        sem = asyncio.Semaphore(max_concurrency)

        async def run_with_sem(
            app_name: str, prompt: str
        ) -> tuple[str, Path | None, Path | None, str | None]:
            async with sem:
                try:
                    app_dir, log_file = await self.generate_single(
                        prompt, app_name, backend, model, mcp_args
                    )
                    return (app_name, app_dir, log_file, None)
                except Exception as e:
                    log_path = self.output_dir / "logs" / f"{app_name}.log"
                    return (app_name, None, log_path if log_path.exists() else None, str(e))

        tasks = [run_with_sem(name, prompt) for name, prompt in prompts.items()]
        return await asyncio.gather(*tasks)

    async def _build_container(self, client: dagger.Client) -> dagger.Container:
        """Build container from Dockerfile with layer caching."""
        # build context excluding generated files
        context = client.host().directory(
            ".",
            exclude=[
                "app/",
                "app-eval/",
                "results/",
                ".venv/",
                "__pycache__/",
                ".git/",
            ],
        )

        # build from Dockerfile (leverages BuildKit cache)
        container = context.docker_build()

        # mount mcp binary from host (not baked into image)
        container = container.with_file(
            "/usr/local/bin/edda_mcp",
            client.host().file(str(self.mcp_binary)),
            permissions=0o755,  # make executable
        )

        # pass through env vars from host
        env_vars = [
            "ANTHROPIC_API_KEY",
            "NEON_DATABASE_URL",
        ]
        for var in env_vars:
            if val := os.environ.get(var):
                container = container.with_env_variable(var, val)

        # mount databricks config for CLI authentication (OAuth profile)
        # container runs as 'klaudbiusz' user (see Dockerfile)
        databrickscfg = Path.home() / ".databrickscfg"
        if databrickscfg.exists():
            container = container.with_file(
                "/home/klaudbiusz/.databrickscfg",
                client.host().file(str(databrickscfg)),
                owner="klaudbiusz:klaudbiusz",
            )

        # mount databricks directory for OAuth token cache and other CLI state
        # required when using auth_type = databricks-cli
        databricks_dir = Path.home() / ".databricks"
        if databricks_dir.exists():
            container = container.with_directory(
                "/home/klaudbiusz/.databricks",
                client.host().directory(str(databricks_dir)),
                owner="klaudbiusz:klaudbiusz",
            )

        return container
