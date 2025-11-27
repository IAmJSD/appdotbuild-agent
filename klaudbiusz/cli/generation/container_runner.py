"""Runner script executed inside Dagger container."""

import json
import sys

import fire


def run(
    prompt: str,
    app_name: str,
    backend: str = "claude",
    model: str | None = None,
    mcp_args: str | list[str] | None = None,
) -> None:
    """Run app generation inside container.

    Args:
        prompt: The prompt describing what to build
        app_name: App name for output directory
        backend: "claude" or "litellm"
        model: Model name (required for litellm)
        mcp_args: JSON-encoded list or already-parsed list of MCP server args
    """
    # handle both JSON string and already-parsed list (fire may parse it)
    parsed_mcp_args: list[str] | None
    match mcp_args:
        case None:
            parsed_mcp_args = None
        case str():
            parsed_mcp_args = json.loads(mcp_args)
        case list():
            parsed_mcp_args = mcp_args

    match backend:
        case "claude":
            from cli.generation.codegen import ClaudeAppBuilder

            builder = ClaudeAppBuilder(
                app_name=app_name,
                wipe_db=False,
                suppress_logs=False,
                mcp_binary="/usr/local/bin/edda_mcp",
                mcp_args=parsed_mcp_args,
                output_dir="/workspace/app",
            )
            metrics = builder.run(prompt, wipe_db=False)
        case "litellm":
            from cli.generation.codegen_multi import LiteLLMAppBuilder

            if not model:
                print("Error: --model is required for litellm backend", file=sys.stderr)
                sys.exit(1)

            builder = LiteLLMAppBuilder(
                app_name=app_name,
                model=model,
                mcp_binary="/usr/local/bin/edda_mcp",
                mcp_args=parsed_mcp_args,
                suppress_logs=False,
                output_dir="/workspace/app",
            )
            metrics = builder.run(prompt)
        case _:
            print(f"Error: Unknown backend: {backend}", file=sys.stderr)
            sys.exit(1)

    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    fire.Fire(run)
