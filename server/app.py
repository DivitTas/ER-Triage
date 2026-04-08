# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Er Triage Environment.

This module creates an HTTP server that exposes the ErTriageEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    # Try relative imports first (for package usage)
    from ..models import ErTriageAction, ErTriageObservation
    from .ER_Triage_environment import ErTriageEnvironment
except (ImportError, ValueError):
    # Docker/direct execution fallback
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from models import ErTriageAction, ErTriageObservation
    from server.ER_Triage_environment import ErTriageEnvironment


# Create the app with web interface and README integration
app = create_app(
    ErTriageEnvironment,
    ErTriageAction,
    ErTriageObservation,
    env_name="ER_Triage",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# Export for openenv validate
__all__ = ["app", "main", "run_server"]


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Run the FastAPI application with explicit host/port values.

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m ER_Triage.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn ER_Triage.server.app:app --workers 4
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
