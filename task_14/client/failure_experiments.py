from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .client import send_request
except ImportError:
    from client import send_request


def run(base_url: str, unavailable_url: str, timeout: float) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    for name, url, path in [
        ("invalid_route", base_url, "/does-not-exist"),
        ("server_unavailable_or_wrong_port", unavailable_url, "/health"),
    ]:
        try:
            status, response = send_request(url, path, timeout=timeout)
            results.append({"experiment": name, "status": status, "response": response})
        except ConnectionError as exc:
            results.append({"experiment": name, "connection_error": str(exc)})

    malformed = Request(
        f"{base_url.rstrip('/')}/echo",
        data=b'{"message":',
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(malformed, timeout=timeout) as response:
            results.append({"experiment": "malformed_json", "status": response.status})
    except HTTPError as exc:
        results.append(
            {
                "experiment": "malformed_json",
                "status": exc.code,
                "response": json.loads(exc.read().decode("utf-8")),
            }
        )
    except URLError as exc:
        results.append({"experiment": "malformed_json", "connection_error": str(exc.reason)})

    wrong_type = Request(
        f"{base_url.rstrip('/')}/echo",
        data=b'["not", "an", "object"]',
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(wrong_type, timeout=timeout) as response:
            results.append({"experiment": "incorrect_payload_type", "status": response.status})
    except HTTPError as exc:
        results.append(
            {
                "experiment": "incorrect_payload_type",
                "status": exc.code,
                "response": json.loads(exc.read().decode("utf-8")),
            }
        )
    except URLError as exc:
        results.append({"experiment": "incorrect_payload_type", "connection_error": str(exc.reason)})

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce representative network- and application-level failures."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--unavailable-url", default="http://127.0.0.1:65534")
    parser.add_argument("--timeout", default=2.0, type=float)
    args = parser.parse_args()
    print(json.dumps(run(args.base_url, args.unavailable_url, args.timeout), indent=2))


if __name__ == "__main__":
    main()
