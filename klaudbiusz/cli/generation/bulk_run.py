"""Bulk app generation via Dagger with parallelism."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import fire
from dotenv import load_dotenv

from cli.generation.dagger_run import DaggerAppGenerator

load_dotenv()


def _restore_terminal_cursor() -> None:
    """Restore terminal cursor after Dagger run (workaround for dagger/dagger#7160)."""
    os.system("tput cnorm 2>/dev/null || true")


def main(
    prompts: str = "databricks",
    backend: str = "claude",
    model: str | None = None,
    mcp_binary: str | None = None,
    mcp_args: list[str] | None = None,
    output_dir: str | None = None,
    max_concurrency: int = 4,
) -> None:
    """Bulk app generation via Dagger with parallelism.

    Args:
        prompts: Prompt set to use ("databricks", "databricks_v2", or "test")
        backend: Backend to use ("claude" or "litellm")
        model: LLM model (required if backend=litellm)
        mcp_binary: Path to edda_mcp binary (required)
        mcp_args: Optional list of args passed to the MCP server
        output_dir: Custom output directory for generated apps
        max_concurrency: Maximum parallel generations (default: 4)

    Usage:
        # Claude backend with databricks prompts
        python bulk_run.py --mcp_binary=/path/to/edda_mcp

        # With custom concurrency
        python bulk_run.py --mcp_binary=/path/to/edda_mcp --max_concurrency=8

        # LiteLLM backend
        python bulk_run.py --backend=litellm --model=gemini/gemini-2.5-pro --mcp_binary=/path/to/edda_mcp
    """
    if not mcp_binary:
        raise ValueError("--mcp_binary is required")

    if backend == "litellm" and not model:
        raise ValueError("--model is required when using --backend=litellm")

    # validate required environment variables
    if not os.environ.get("DATABRICKS_HOST") or not os.environ.get("DATABRICKS_TOKEN"):
        raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN environment variables must be set")

    # load prompt set
    match prompts:
        case "databricks":
            from cli.generation.prompts.databricks import PROMPTS as selected_prompts
        case "databricks_v2":
            from cli.generation.prompts.databricks_v2 import PROMPTS as selected_prompts
        case "test":
            from cli.generation.prompts.web import PROMPTS as selected_prompts
        case _:
            raise ValueError(f"Unknown prompt set: {prompts}. Use 'databricks', 'databricks_v2', or 'test'")

    print(f"Starting bulk generation for {len(selected_prompts)} prompts...")
    print(f"Backend: {backend}")
    if backend == "litellm":
        print(f"Model: {model}")
    print(f"Prompt set: {prompts}")
    print(f"Max concurrency: {max_concurrency}")
    print(f"MCP binary: {mcp_binary}")
    out_path = Path(output_dir) if output_dir else Path("./app")
    print(f"Output dir: {out_path}\n")

    generator = DaggerAppGenerator(
        mcp_binary=Path(mcp_binary),
        output_dir=out_path,
    )

    try:
        results = asyncio.run(
            generator.generate_bulk(
                selected_prompts,
                backend,
                model,
                mcp_args,
                max_concurrency,
            )
        )
    finally:
        _restore_terminal_cursor()

    # separate successful and failed
    successful = [(name, app_dir, log) for name, app_dir, log, err in results if err is None]
    failed = [(name, log, err) for name, app_dir, log, err in results if err is not None]

    print(f"\n{'=' * 80}")
    print("Bulk Generation Summary")
    print(f"{'=' * 80}")
    print(f"Total prompts: {len(selected_prompts)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\n{'=' * 80}")
        print("Failed generations:")
        print(f"{'=' * 80}")
        for name, log, err in failed:
            print(f"  - {name}")
            print(f"    Error: {err}")
            if log:
                print(f"    Log: {log}")

    if successful:
        print(f"\n{'=' * 80}")
        print("Generated apps:")
        print(f"{'=' * 80}")
        for name, app_dir, log in successful:
            print(f"  - {name}")
            print(f"    Dir: {app_dir}")
            print(f"    Log: {log}")

    print(f"\n{'=' * 80}\n")

    # save results json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_suffix = f"_{backend}" if backend != "claude" else ""
    output_file = out_path / f"bulk_run_results{backend_suffix}_{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results_data = [
        {
            "app_name": name,
            "success": err is None,
            "app_dir": str(app_dir) if app_dir else None,
            "log_file": str(log) if log else None,
            "error": err,
            "backend": backend,
            "model": model,
        }
        for name, app_dir, log, err in results
    ]
    output_file.write_text(json.dumps(results_data, indent=2))
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
