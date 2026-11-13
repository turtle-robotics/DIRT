"""DIRT distance sensor support (HC-SR04 ultrasonic).

``DistanceSensor`` picks a backend automatically: a mock sensor in
simulation, or the real sensor through ``gpiozero`` on a Pi. If the GPIO
sensor fails to start, it falls back to the mock and logs a warning.

Legacy: the HC-SR04 is from the earlier concept and is not in the PDR design.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

from config import CFG, Platform

log = logging.getLogger(__name__)


@dataclass
class DistanceReading:
    """One distance sample.

    Attributes:
        timestamp_s: Seconds since the start of the session.
        distance_cm: Measured distance in centimetres.
        displacement_cm: Change since the previous sample (0 for a single sample).
    """

    timestamp_s: float
    distance_cm: float
    displacement_cm: float


@dataclass
class SampleSession:
    """A list of readings from ``DistanceSensor.capture_session``."""

    readings: List[DistanceReading] = field(default_factory=list)

    def summary(self) -> dict:
        """Return count, duration and min/max/mean/first/last distance as a dict."""
        if not self.readings:
            return {"n": 0}
        distances = [r.distance_cm for r in self.readings]
        return {
            "n": len(distances),
            "duration_s": self.readings[-1].timestamp_s - self.readings[0].timestamp_s,
            "mean_cm": round(sum(distances) / len(distances), 2),
            "min_cm": round(min(distances), 2),
            "max_cm": round(max(distances), 2),
            "range_cm": round(max(distances) - min(distances), 2),
            "first_cm": round(distances[0], 2),
            "last_cm": round(distances[-1], 2),
        }

    def to_csv(self) -> str:
        """Return the readings as CSV text with a header row."""
        lines = ["time_s,distance_cm,displacement_cm"]
        for r in self.readings:
            lines.append(f"{r.timestamp_s:.3f},{r.distance_cm:.3f},{r.displacement_cm:.3f}")
        return "\n".join(lines)


class _SensorBackend:
    """Interface for a sensor backend. ``read_cm`` returns one distance in cm."""

    def read_cm(self) -> float:
        """Return one distance reading in centimetres."""
        raise NotImplementedError


class _GPIOSensor(_SensorBackend):
    """Real HC-SR04 read through ``gpiozero`` (Pi only)."""

    def __init__(self, cfg=CFG.distance):
        """Create the gpiozero sensor.

        Args:
            cfg: Pins and range. Defaults to ``CFG.distance``.
        """
        from gpiozero import DistanceSensor as _DS

        self._sensor = _DS(
            echo=cfg.echo_pin,
            trigger=cfg.trig_pin,
            max_distance=cfg.max_distance_cm / 100.0,
        )
        log.info("[HC-SR04] GPIO sensor init: trig=%d echo=%d", cfg.trig_pin, cfg.echo_pin)

    def read_cm(self) -> float:
        """Return the distance in cm (gpiozero reports metres)."""
        return self._sensor.distance * 100.0


class _MockSensor(_SensorBackend):
    """Fake sensor for testing without hardware.

    Starts at 150 cm and drifts toward the sensor by about 0.3 cm per read, with
    Gaussian noise (sigma 0.5 cm), clamped to 10..200 cm. That looks roughly like
    something slowly approaching.
    """

    def __init__(self):
        """Start at 150 cm with a small negative drift."""
        self._pos = 150.0
        self._drift = -0.3
        log.info("[HC-SR04] Mock sensor active (no hardware)")

    def read_cm(self) -> float:
        """Return the next simulated distance in cm."""
        self._pos += self._drift + random.gauss(0, 0.5)
        self._pos = max(10.0, min(200.0, self._pos))
        return self._pos


def _make_backend() -> _SensorBackend:
    """Return the mock in simulation, otherwise try the GPIO sensor and fall back to the mock.
    """
    if CFG.platform == Platform.SIMULATION:
        return _MockSensor()
    try:
        return _GPIOSensor()
    except Exception as exc:
        log.warning("GPIO sensor failed (%s) - falling back to mock", exc)
        return _MockSensor()


class DistanceSensor:
    """Public distance sensor API.

    Example::

        with DistanceSensor() as ds:
            session = ds.capture_session(duration_s=5)
            print(session.summary())
    """

    def __init__(self, cfg=CFG.distance):
        """Pick a backend and store the settings.

        Args:
            cfg: Sensor settings. Defaults to ``CFG.distance``.
        """
        self.cfg = cfg
        self._backend = _make_backend()

    def read_cm(self) -> float:
        """Return one reading in centimetres."""
        return self._backend.read_cm()

    def read_distance_m(self) -> float:
        """Return one reading in metres."""
        return self.read_cm() / 100.0

    def sample(self, time_s: float) -> DistanceReading:
        """Take one reading and stamp it with the given time.

        Args:
            time_s: Timestamp to store with the reading.
        """
        distance_cm = self.read_cm()
        return DistanceReading(time_s, distance_cm, 0.0)

    def capture_session(self, duration_s: Optional[float] = None, sample_hz: Optional[float] = None) -> SampleSession:
        """Record readings for a fixed time and return them.

        Prints a "Ready / Set / Go" countdown (about 4 s) before recording, then
        prints each reading as it is taken.

        Args:
            duration_s: How long to record. Defaults to ``cfg.capture_duration_s``.
            sample_hz: Readings per second. Defaults to ``cfg.sample_rate_hz``.

        Returns:
            A ``SampleSession`` with one ``DistanceReading`` per sample.
        """
        duration_s = duration_s or self.cfg.capture_duration_s
        sample_hz = sample_hz or self.cfg.sample_rate_hz
        interval = 1.0 / sample_hz
        session = SampleSession()

        print("Ready")
        time.sleep(2.0)
        print("Set")
        time.sleep(1.0)
        print("Go")
        time.sleep(1.0)

        start = time.monotonic()
        prev_cm = self.read_cm()

        while True:
            elapsed = time.monotonic() - start
            if elapsed >= duration_s:
                break

            dist_cm = self.read_cm()
            disp_cm = dist_cm - prev_cm
            prev_cm = dist_cm

            reading = DistanceReading(elapsed, dist_cm, disp_cm)
            session.readings.append(reading)
            print(f"Time(s):{elapsed:.3f}  Distance(m):{dist_cm/100.0:.3f}  Displacement(cm):{disp_cm:+.3f}")

            time.sleep(interval)

        log.info("Session complete: %s", session.summary())
        return session

    def __enter__(self):
        """Allow use as a context manager."""
        return self

    def __exit__(self, *_):
        """Nothing to clean up; present so ``with`` works."""
        pass
