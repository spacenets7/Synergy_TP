# Task 14 - Local Networking and API Communication

Task 14 implements a small FastAPI service and a separate Python client for learning HTTP and JSON communication on one machine or across a local network.

## API Routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Check whether the service is running |
| `GET` | `/info` | Return service, protocol, client, and route information |
| `POST` | `/echo` | Validate and return a JSON object with request metadata |

Interactive API documentation is available at `/docs` while the server is running.

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r task_14\requirements.txt
```

## Run the Server

For same-device access only:

```powershell
python -m task_14.server.run
```

For another device on the same LAN:

```powershell
python -m task_14.server.run --host 0.0.0.0 --port 8000
```

Clients must connect to the server computer's LAN IPv4 address, such as `http://192.168.1.10:8000`. They must not use `0.0.0.0` as the destination address.

## Run the Client

```powershell
python -m task_14.client.client health
python -m task_14.client.client info
python -m task_14.client.client echo
python -m task_14.client.client echo --payload '{"message":"hello","sequence":2}'
```

For a remote server:

```powershell
python -m task_14.client.client health --base-url http://192.168.1.10:8000
```

The client prints the HTTP status and decoded response. Connection errors identify the address, port, binding, server-process, and firewall checks that may resolve the failure.

## Failure Experiments

Keep the normal server running, then execute:

```powershell
python -m task_14.client.failure_experiments
```

The script reproduces four cases: an invalid route, an unavailable service or wrong port, malformed JSON, and an incorrect payload type.

## Tests

```powershell
pytest task_14\tests -v
```

The automated tests cover all required routes and application-level validation. Cross-device reachability must still be demonstrated from a second physical device on the same network.

## Structure

```text
task_14/
|-- client/
|   |-- client.py
|   `-- failure_experiments.py
|-- server/
|   |-- app.py
|   `-- run.py
|-- tests/
|   `-- test_api.py
`-- requirements.txt
```
