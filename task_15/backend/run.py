from __future__ import annotations

import argparse
import logging

import uvicorn

try:
    from .app import app
except ImportError:
    from app import app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the wearable-vitals ingestion backend.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Use 0.0.0.0 to accept LAN connections.",
    )
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.reload:
        application = f"{__package__}.app:app" if __package__ else "app:app"
    else:
        application = app
    uvicorn.run(
        application,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
