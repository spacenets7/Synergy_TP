from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SignalQuality(str, Enum):
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class DeviceState(str, Enum):
    ACTIVE = "active"
    LOW_BATTERY = "low_battery"
    SENSOR_ERROR = "sensor_error"


class VitalReadings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heart_rate_bpm: float = Field(ge=25.0, le=240.0, allow_inf_nan=False)
    spo2_percent: float = Field(ge=70.0, le=100.0, allow_inf_nan=False)
    body_temperature_c: float = Field(ge=30.0, le=43.0, allow_inf_nan=False)


class DeviceStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    battery_percent: float = Field(ge=0.0, le=100.0, allow_inf_nan=False)
    signal_quality: SignalQuality
    state: DeviceState = DeviceState.ACTIVE


class SensorPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]+$")
    session_id: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]+$")
    timestamp: datetime
    profile: str = Field(default="wearable_vitals_v1", pattern=r"^wearable_vitals_v1$")
    readings: VitalReadings
    status: DeviceStatus

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a UTC offset, for example Z or +05:30")
        return value


class StoredReading(SensorPacket):
    assessment: str
    accepted_at: datetime


class AcceptedReading(BaseModel):
    status: str
    reading: StoredReading


class DeviceSummary(BaseModel):
    device_id: str
    reading_count: int
    latest_timestamp: datetime
    latest_assessment: str
