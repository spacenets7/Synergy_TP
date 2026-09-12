from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from task_15.backend.app import app, store

client = TestClient(app)


def valid_packet(device_id: str = "wearable-001", minute: int = 30) -> dict[str, object]:
    return {
        "device_id": device_id,
        "session_id": "session-001",
        "timestamp": f"2026-09-09T10:{minute:02d}:00Z",
        "profile": "wearable_vitals_v1",
        "readings": {
            "heart_rate_bpm": 76.0,
            "spo2_percent": 98.0,
            "body_temperature_c": 36.7,
        },
        "status": {
            "battery_percent": 82.0,
            "signal_quality": "good",
            "state": "active",
        },
    }


@pytest.fixture(autouse=True)
def empty_store() -> None:
    store.clear()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["stored_readings"] == 0


def test_submit_and_retrieve_latest_reading() -> None:
    packet = valid_packet()
    accepted = client.post("/readings", json=packet)
    assert accepted.status_code == 201
    assert accepted.json()["reading"]["assessment"] == "within_configured_range"

    latest = client.get("/readings/latest")
    assert latest.status_code == 200
    assert latest.json()["device_id"] == packet["device_id"]


def test_multiple_readings_and_two_devices() -> None:
    assert client.post("/readings", json=valid_packet("wearable-001", 30)).status_code == 201
    assert client.post("/readings", json=valid_packet("wearable-001", 31)).status_code == 201
    assert client.post("/readings", json=valid_packet("wearable-002", 32)).status_code == 201

    devices = client.get("/devices").json()
    assert [item["device_id"] for item in devices] == ["wearable-001", "wearable-002"]
    assert devices[0]["reading_count"] == 2
    assert devices[1]["reading_count"] == 1

    latest = client.get("/devices/wearable-001/latest")
    assert latest.status_code == 200
    assert latest.json()["timestamp"] == "2026-09-09T10:31:00Z"


def test_abnormal_but_plausible_reading_is_flagged() -> None:
    packet = valid_packet()
    packet["readings"]["spo2_percent"] = 89.0
    response = client.post("/readings", json=packet)
    assert response.status_code == 201
    assert response.json()["reading"]["assessment"] == "critical_reading"


@pytest.mark.parametrize(
    ("mutator", "expected_fragment"),
    [
        (lambda packet: packet.pop("device_id"), "device_id"),
        (lambda packet: packet["readings"].update({"heart_rate_bpm": "fast"}), "heart_rate_bpm"),
        (lambda packet: packet["readings"].update({"spo2_percent": 140.0}), "spo2_percent"),
        (lambda packet: packet.update({"timestamp": "2026-09-09 10:30:00"}), "timestamp"),
        (lambda packet: packet.update({"profile": "unknown_profile"}), "profile"),
    ],
)
def test_invalid_packets_are_rejected(mutator, expected_fragment: str) -> None:
    packet = deepcopy(valid_packet())
    mutator(packet)
    response = client.post("/readings", json=packet)
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    assert expected_fragment in str(response.json()["details"])


def test_missing_latest_returns_clear_404() -> None:
    response = client.get("/readings/latest")
    assert response.status_code == 404
    assert response.json()["detail"] == "No readings have been accepted yet."


def test_missing_device_returns_clear_404() -> None:
    response = client.get("/devices/does-not-exist/latest")
    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]
