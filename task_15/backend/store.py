from __future__ import annotations

from collections import deque
from threading import RLock

from .models import DeviceSummary, StoredReading


class ReadingStore:
    def __init__(self, max_readings: int = 1_000) -> None:
        if max_readings < 1:
            raise ValueError("max_readings must be positive")
        self._readings: deque[StoredReading] = deque(maxlen=max_readings)
        self._lock = RLock()

    def add(self, reading: StoredReading) -> None:
        with self._lock:
            self._readings.append(reading)

    def latest(self) -> StoredReading | None:
        with self._lock:
            return self._readings[-1] if self._readings else None

    def latest_for_device(self, device_id: str) -> StoredReading | None:
        with self._lock:
            for reading in reversed(self._readings):
                if reading.device_id == device_id:
                    return reading
        return None

    def device_summaries(self) -> list[DeviceSummary]:
        with self._lock:
            counts: dict[str, int] = {}
            latest: dict[str, StoredReading] = {}
            for reading in self._readings:
                counts[reading.device_id] = counts.get(reading.device_id, 0) + 1
                latest[reading.device_id] = reading

        return [
            DeviceSummary(
                device_id=device_id,
                reading_count=counts[device_id],
                latest_timestamp=reading.timestamp,
                latest_assessment=reading.assessment,
            )
            for device_id, reading in sorted(latest.items())
        ]

    def clear(self) -> None:
        with self._lock:
            self._readings.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._readings)
