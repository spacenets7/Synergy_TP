import random
from urllib.error import URLError

from task_15.simulator import simulator


def test_normal_packet_matches_contract_shape() -> None:
    packet = simulator.generate_packet(random.Random(42), "wearable-001", "session-001")
    assert packet["profile"] == "wearable_vitals_v1"
    assert 95.0 <= packet["readings"]["spo2_percent"] <= 100.0
    assert packet["device_id"] == "wearable-001"


def test_abnormal_packet_remains_physically_plausible() -> None:
    packet = simulator.generate_packet(
        random.Random(42), "wearable-001", "session-001", "abnormal"
    )
    assert packet["readings"]["spo2_percent"] == 89.0


def test_invalid_packet_exceeds_contract_limit() -> None:
    packet = simulator.generate_packet(
        random.Random(42), "wearable-001", "session-001", "invalid"
    )
    assert packet["readings"]["spo2_percent"] == 140.0


def test_unavailable_backend_returns_failure(monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr(simulator, "urlopen", unavailable)
    packet = simulator.generate_packet(random.Random(42), "wearable-001", "session-001")
    result = simulator.send_packet("http://127.0.0.1:65534", packet, timeout=0.01)
    assert result.success is False
    assert result.status_code is None
    assert "Backend unavailable" in result.response
