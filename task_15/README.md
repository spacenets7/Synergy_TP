# Task 15 - MedTech Sensor Data Ingestion

Task 15 simulates a wearable vital-sign monitor and sends structured readings to a FastAPI backend. The backend validates each packet, assigns a non-diagnostic assessment, keeps the latest 1,000 accepted readings in memory, and supports overall and device-specific retrieval.

## Sensor Contract

Each `wearable_vitals_v1` packet contains:

- `device_id` and `session_id`
- A timezone-aware ISO 8601 timestamp
- Heart rate in beats per minute
- Blood-oxygen saturation as a percentage
- Body temperature in degrees Celsius
- Battery level, signal quality, and device state

Physically plausible acceptance limits are 25-240 bpm, 70-100% SpO2, and 30-43 degrees Celsius. Unusual but plausible readings are accepted and flagged; values outside these limits receive an HTTP 422 response. Assessments are demonstration rules and are not medical diagnoses.

## API Routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Service status and stored-reading count |
| `POST` | `/readings` | Validate and store one sensor packet |
| `GET` | `/readings/latest` | Retrieve the latest accepted reading |
| `GET` | `/devices` | List known devices and reading counts |
| `GET` | `/devices/{device_id}/latest` | Retrieve one device's latest reading |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r task_15\requirements.txt
```

## Run the Backend

Same-device access:

```powershell
python -m task_15.backend.run
```

LAN access from another device:

```powershell
python -m task_15.backend.run --host 0.0.0.0 --port 8001
```

The OpenAPI interface is available at `http://127.0.0.1:8001/docs`. A remote simulator must replace `127.0.0.1` with the backend computer's LAN IPv4 address.

## Run the Simulator

The default mixed run sends normal readings, one abnormal but plausible reading, and one deliberately invalid packet:

```powershell
python -m task_15.simulator.simulator
```

Useful options:

```powershell
python -m task_15.simulator.simulator --mode normal --count 20 --interval 0.5
python -m task_15.simulator.simulator --mode abnormal --count 1
python -m task_15.simulator.simulator --mode invalid --count 1
python -m task_15.simulator.simulator --device-id wearable-002 --session-id trial-002
python -m task_15.simulator.simulator --base-url http://192.168.1.10:8001
```

The simulator logs each request, accepted response, validation failure, and connection failure. A mixed or invalid run returns a nonzero exit status because at least one packet is intentionally rejected.

## Payload Examples and Tests

Valid, abnormal, missing-field, wrong-type, and out-of-range examples are available in `examples/`.

```powershell
pytest task_15\tests -v
```

The tests cover validation, two-device storage, latest-reading retrieval, abnormal-reading assessment, invalid profiles and timestamps, and unavailable-backend handling. Physical cross-device operation must be verified on the target LAN.

## Structure

```text
task_15/
|-- backend/       # FastAPI routes, schemas, assessment, and in-memory store
|-- simulator/     # Configurable wearable device simulator
|-- examples/      # Valid and invalid JSON packets
|-- tests/         # API and simulator tests
`-- requirements.txt
```
