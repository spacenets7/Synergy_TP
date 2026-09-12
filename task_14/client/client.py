from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def send_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> tuple[int, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{base_url.rstrip('/')}/{path.lstrip('/')}",
        data=body,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, _decode_response(raw)
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, _decode_response(raw)
    except URLError as exc:
        raise ConnectionError(
            f"Could not connect to {request.full_url}: {exc.reason}. "
            "Check the server process, IP address, port, binding, and firewall."
        ) from exc


def _decode_response(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_payload(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("The echo payload must be a JSON object.")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the Task 14 HTTP service.")
    parser.add_argument("command", choices=["health", "info", "echo"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--payload",
        type=parse_payload,
        default={"message": "hello from task 14", "sequence": 1},
        help="JSON object used by the echo command.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    route = {"health": "/health", "info": "/info", "echo": "/echo"}[args.command]
    method = "POST" if args.command == "echo" else "GET"
    payload = args.payload if method == "POST" else None
    try:
        status, response = send_request(
            args.base_url,
            route,
            method=method,
            payload=payload,
            timeout=args.timeout,
        )
    except ConnectionError as exc:
        print(f"CONNECTION ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"HTTP {status}")
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0 if 200 <= status < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
