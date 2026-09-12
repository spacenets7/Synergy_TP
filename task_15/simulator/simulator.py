from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("task_15.simulator")


@dataclass(frozen=True)
class RequestResult:
    success: bool
    status_code: int | None
    response: Any


def generate_packet(
    rng: random.Random,
    device_id: str,
    session_id: str,
    mode: str = "normal",
) -> dict[str, Any]:
    readings = {
        "heart_rate_bpm": round(rng.gauss(76.0, 7.0), 1),
        "spo2_percent": round(min(100.0, max(95.0, rng.gauss(97.6, 0.8))), 1),
        "body_temperature_c": round(rng.gauss(36.7, 0.25), 1),
    }
    status = {
        "battery_percent": round(rng.uniform(35.0, 100.0), 1),
        "signal_quality": rng.choice(["good", "good", "good", "fair"]),
        "state": "active",
    }

    if mode == "abnormal":
        readings.update({"heart_rate_bpm": 132.0, "spo2_percent": 89.0})
    elif mode == "invalid":
        readings["spo2_percent"] = 140.0
    elif mode != "normal":
        raise ValueError(f"Unsupported generation mode: {mode}")

    return {
        "device_id": device_id,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": "wearable_vitals_v1",
        "readings": readings,
        "status": status,
    }


def send_packet(base_url: str, packet: dict[str, Any], timeout: float = 5.0) -> RequestResult:
    request = Request(
        f"{base_url.rstrip('/')}/readings",
        data=json.dumps(packet).encode("utf-8"),
        method="POST",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    logger.info("POST %s device=%s session=%s", request.full_url, packet.get("device_id"), packet.get("session_id"))
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            logger.info("Backend response HTTP %s status=%s", response.status, payload.get("status"))
            return RequestResult(True, response.status, payload)
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
        logger.warning("Backend rejected packet HTTP %s response=%s", exc.code, payload)
        return RequestResult(False, exc.code, payload)
    except URLError as exc:
        message = (
            f"Backend unavailable at {request.full_url}: {exc.reason}. "
            "The simulator will continue unless all configured packets have been attempted."
        )
        logger.error(message)
        return RequestResult(False, None, message)


def select_mode(requested_mode: str, index: int, count: int) -> str:
    if requested_mode != "mixed":
        return requested_mode
    if count == 1:
        return "abnormal"
    if index == count - 2:
        return "abnormal"
    if index == count - 1:
        return "invalid"
    return "normal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a wearable vital-sign monitor.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--device-id", default="wearable-001")
    parser.add_argument("--session-id", default="session-001")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["normal", "abnormal", "invalid", "mixed"], default="mixed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if args.interval < 0:
        raise SystemExit("--interval cannot be negative")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    rng = random.Random(args.seed)
    successes = 0
    for index in range(args.count):
        mode = select_mode(args.mode, index, args.count)
        packet = generate_packet(rng, args.device_id, args.session_id, mode)
        result = send_packet(args.base_url, packet, args.timeout)
        successes += int(result.success)
        print(
            json.dumps(
                {
                    "sequence": index + 1,
                    "mode": mode,
                    "success": result.success,
                    "status_code": result.status_code,
                    "response": result.response,
                },
                indent=2,
            )
        )
        if index < args.count - 1:
            time.sleep(args.interval)

    logger.info("Simulation complete: %d/%d packets accepted", successes, args.count)
    return 0 if successes == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
