from fastapi.testclient import TestClient

from task_14.server.app import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_info_lists_required_routes() -> None:
    response = client.get("/info")
    assert response.status_code == 200
    routes = response.json()["routes"]
    assert routes == {
        "health": "GET /health",
        "info": "GET /info",
        "echo": "POST /echo",
    }


def test_echo_returns_payload_and_metadata() -> None:
    payload = {"message": "hello", "sequence": 7}
    response = client.post("/echo", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert data["received"] == payload
    assert data["field_count"] == 2


def test_echo_rejects_non_object_json() -> None:
    response = client.post("/echo", json=["invalid"])
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_unknown_route_returns_404() -> None:
    response = client.get("/not-a-route")
    assert response.status_code == 404
