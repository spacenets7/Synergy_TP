from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import RootModel

logger = logging.getLogger("task_14.server")


class EchoPayload(RootModel[dict[str, Any]]):
    pass


app = FastAPI(
    title="Task 14 Local Network API",
    version="1.0.0",
    description="A small service for testing HTTP and JSON communication across a LAN.",
)


@app.get("/health", tags=["service"])
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": app.title,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/info", tags=["service"])
def info(request: Request) -> dict[str, Any]:
    client = request.client
    return {
        "name": app.title,
        "version": app.version,
        "protocol": "HTTP over TCP",
        "client": {
            "host": client.host if client else None,
            "port": client.port if client else None,
        },
        "routes": {
            "health": "GET /health",
            "info": "GET /info",
            "echo": "POST /echo",
        },
    }


@app.post("/echo", tags=["communication"])
def echo(payload: EchoPayload, request: Request) -> dict[str, Any]:
    data = payload.root
    logger.info("Echo accepted from %s with %d fields", request.client, len(data))
    return {
        "status": "accepted",
        "received": data,
        "field_count": len(data),
        "received_at": datetime.now(timezone.utc).isoformat(),
    }


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning("Validation failed for %s %s: %s", request.method, request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "The request body must be a valid JSON object.",
            "details": exc.errors(),
        },
    )
