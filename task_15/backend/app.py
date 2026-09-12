from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .models import AcceptedReading, DeviceSummary, SensorPacket, StoredReading
from .store import ReadingStore

logger = logging.getLogger("task_15.backend")
store = ReadingStore(max_readings=1_000)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("Task 15 sensor-ingestion backend started")
    yield
    logger.info("Task 15 sensor-ingestion backend stopped")


app = FastAPI(
    title="Task 15 Wearable Vitals API",
    version="1.0.0",
    description="Validates and temporarily stores readings from simulated wearable monitors.",
    lifespan=lifespan,
)


def assess(packet: SensorPacket) -> str:
    reading = packet.readings
    if packet.status.state == "sensor_error" or packet.status.signal_quality == "poor":
        return "sensor_warning"
    if (
        reading.heart_rate_bpm < 40
        or reading.heart_rate_bpm > 180
        or reading.spo2_percent < 90
        or reading.body_temperature_c < 34.0
        or reading.body_temperature_c > 40.0
    ):
        return "critical_reading"
    if (
        reading.heart_rate_bpm < 50
        or reading.heart_rate_bpm > 110
        or reading.spo2_percent < 95
        or reading.body_temperature_c < 35.5
        or reading.body_temperature_c > 37.5
    ):
        return "review_recommended"
    return "within_configured_range"


@app.get("/health", tags=["service"])
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": app.title,
        "stored_readings": len(store),
        "timestamp": datetime.now(timezone.utc),
    }


@app.post(
    "/readings",
    response_model=AcceptedReading,
    status_code=status.HTTP_201_CREATED,
    tags=["readings"],
)
def submit_reading(packet: SensorPacket) -> AcceptedReading:
    stored = StoredReading(
        **packet.model_dump(),
        assessment=assess(packet),
        accepted_at=datetime.now(timezone.utc),
    )
    store.add(stored)
    logger.info(
        "Accepted reading device=%s session=%s assessment=%s",
        stored.device_id,
        stored.session_id,
        stored.assessment,
    )
    return AcceptedReading(status="accepted", reading=stored)


@app.get("/readings/latest", response_model=StoredReading, tags=["readings"])
def latest_reading() -> StoredReading:
    reading = store.latest()
    if reading is None:
        raise HTTPException(status_code=404, detail="No readings have been accepted yet.")
    return reading


@app.get("/devices", response_model=list[DeviceSummary], tags=["devices"])
def devices() -> list[DeviceSummary]:
    return store.device_summaries()


@app.get("/devices/{device_id}/latest", response_model=StoredReading, tags=["devices"])
def latest_for_device(device_id: str) -> StoredReading:
    reading = store.latest_for_device(device_id)
    if reading is None:
        raise HTTPException(status_code=404, detail=f"No readings found for device '{device_id}'.")
    return reading


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details = jsonable_encoder(exc.errors())
    logger.warning("Validation failed for %s %s: %s", request.method, request.url.path, details)
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "The sensor packet does not match the wearable_vitals_v1 contract.",
            "details": details,
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unexpected processing error for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "The request could not be processed."},
    )
