"""DIRT Bot navigation stack."""

from __future__ import annotations

import csv
import logging
import math
import random
import time
from importlib import import_module
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from config import CFG, Platform

log = logging.getLogger(__name__)


@dataclass
class GPSCoord:
    lat: float
    lon: float
    alt: float = 0.0
    hdop: float = 99.9
    fix: bool = False

    def distance_to(self, other: "GPSCoord") -> float:
        """Haversine distance in metres."""
        radius = 6_371_000.0
        phi1 = math.radians(self.lat)
        phi2 = math.radians(other.lat)
        d_phi = math.radians(other.lat - self.lat)
        d_lambda = math.radians(other.lon - self.lon)
        a = (
            math.sin(d_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        )
        return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def bearing_to(self, other: "GPSCoord") -> float:
        """Bearing in degrees (0 = North, 90 = East)."""
        d_lambda = math.radians(other.lon - self.lon)
        phi1 = math.radians(self.lat)
        phi2 = math.radians(other.lat)
        x = math.sin(d_lambda) * math.cos(phi2)
        y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
        return (math.degrees(math.atan2(x, y)) + 360) % 360


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


@dataclass
class Waypoint:
    name: str
    gps: GPSCoord
    local: Optional[Pose2D] = None
    drilled: bool = False
    measured: bool = False
    action: str = "sample"
    tolerance_m: float = 0.5

    @classmethod
    def from_csv_row(cls, row: dict) -> "Waypoint":
        gps = GPSCoord(
            lat=float(row.get("lat") or row.get("latitude") or 0.0),
            lon=float(row.get("lon") or row.get("lng") or row.get("longitude") or 0.0),
        )
        return cls(
            name=str(row.get("name") or "Waypoint"),
            gps=gps,
            action=str(row.get("action") or "sample"),
            tolerance_m=float(row.get("tolerance_m") or row.get("tolerance") or 0.5),
        )


class NavState(Enum):
    IDLE = auto()
    GPS_NAVIGATE = auto()
    SLAM_LOCALISE = auto()
    ALIGNING = auto()
    DRILLING = auto()
    MEASURING = auto()
    DONE = auto()
    ERROR = auto()


class GPSReceiver:
    """Reads NMEA sentences from a serial GPS module or uses a mock fallback."""

    def __init__(self, cfg=CFG.navigation):
        self._cfg = cfg
        self._coord = GPSCoord(0, 0, fix=False)
        self._serial = None
        if CFG.platform != Platform.SIMULATION:
            self._serial = self._connect()

    def _connect(self):
        try:
            serial_mod = import_module("serial")
            serial_port = serial_mod.Serial(self._cfg.gps_port, self._cfg.gps_baudrate, timeout=1.0)
            log.info("[GPS] Connected on %s @ %d baud", self._cfg.gps_port, self._cfg.gps_baudrate)
            return serial_port
        except Exception as exc:
            log.warning("[GPS] Serial failed (%s) - using mock", exc)
            return None

    def read(self) -> GPSCoord:
        if self._serial is None:
            return self._mock_read()
        return self._parse_serial()

    def _parse_serial(self) -> GPSCoord:
        serial_port = self._serial
        if serial_port is None:
            return self._mock_read()

        try:
            line = serial_port.readline().decode("ascii", errors="replace").strip()
            if line.startswith("$GPRMC"):
                parts = line.split(",")
                if len(parts) > 6 and parts[2] == "A":
                    lat = self._nmea_to_dd(parts[3], parts[4])
                    lon = self._nmea_to_dd(parts[5], parts[6])
                    self._coord = GPSCoord(lat, lon, fix=True)
            elif line.startswith("$GPGGA"):
                parts = line.split(",")
                if len(parts) > 8 and int(parts[6] or 0) > 0:
                    lat = self._nmea_to_dd(parts[2], parts[3])
                    lon = self._nmea_to_dd(parts[4], parts[5])
                    hdop = float(parts[8] or 99)
                    self._coord = GPSCoord(lat, lon, fix=True, hdop=hdop)
        except Exception as exc:
            log.debug("GPS parse error: %s", exc)
        return self._coord

    @staticmethod
    def _nmea_to_dd(value: str, direction: str) -> float:
        if not value:
            return 0.0
        dot = value.index(".")
        degrees = float(value[: dot - 2])
        minutes = float(value[dot - 2 :]) / 60.0
        decimal_degrees = degrees + minutes
        return -decimal_degrees if direction in ("S", "W") else decimal_degrees

    def _mock_read(self) -> GPSCoord:
        self._coord.lat += 0.00001 * (1 + (0.1 - 0.2 * random.random()))
        self._coord.lon += 0.00001 * (1 + (0.1 - 0.2 * random.random()))
        self._coord.fix = True
        self._coord.hdop = 1.2
        return self._coord


class SLAMInterface:
    def __init__(self):
        self._pose = Pose2D()
        self._ros = self._try_ros()

    def _try_ros(self):
        try:
            import_module("rclpy")
            import_module("geometry_msgs.msg")
            log.info("[SLAM] ROS2 available - using slam_toolbox")
            return True
        except ImportError:
            log.info("[SLAM] ROS2 not available - using dead-reckoning fallback")
            return False

    def get_pose(self) -> Pose2D:
        if self._ros:
            pass
        return self._pose

    def update_odometry(self, dx: float, dy: float, dyaw: float):
        yaw_rad = math.radians(self._pose.yaw)
        self._pose.x += dx * math.cos(yaw_rad) - dy * math.sin(yaw_rad)
        self._pose.y += dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
        self._pose.yaw = (self._pose.yaw + dyaw) % 360

    def distance_to_local(self, target: Pose2D) -> float:
        return math.hypot(target.x - self._pose.x, target.y - self._pose.y)

    def bearing_to_local(self, target: Pose2D) -> float:
        return math.degrees(math.atan2(target.x - self._pose.x, target.y - self._pose.y))


class Navigator:
    """High-level waypoint sequencer with compatibility helpers for the simple target API."""

    def __init__(
        self,
        waypoints: Optional[List[Waypoint]] = None,
        gps: Optional[GPSReceiver] = None,
        slam: Optional[SLAMInterface] = None,
    ):
        self.waypoints = waypoints or []
        self.gps = gps or GPSReceiver()
        self.slam = slam or SLAMInterface()
        self.state = NavState.IDLE
        self.current_wp_idx = 0
        self.current_target: Optional[str] = None

    @staticmethod
    def from_csv(path: str) -> "Navigator":
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No waypoint rows found in {path}")
        waypoints = [Waypoint.from_csv_row(row) for row in rows]
        return Navigator(waypoints)

    def status(self) -> dict:
        current = self.current_waypoint
        return {
            "state": self.state.name,
            "waypoint_index": self.current_wp_idx,
            "current_waypoint": current.name if current else None,
            "remaining": max(0, len(self.waypoints) - self.current_wp_idx),
        }

    def set_target(self, target: str) -> None:
        self.current_target = target

    def get_target(self) -> str | None:
        return self.current_target

    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        if self.current_wp_idx < len(self.waypoints):
            return self.waypoints[self.current_wp_idx]
        return None

    def step(self) -> NavState:
        wp = self.current_waypoint
        if wp is None:
            self.state = NavState.DONE
            return self.state

        current_gps = self.gps.read()

        if self.state == NavState.IDLE:
            log.info("Navigating to waypoint: %s", wp.name)
            self.state = NavState.GPS_NAVIGATE

        elif self.state == NavState.GPS_NAVIGATE:
            dist = current_gps.distance_to(wp.gps)
            log.debug("[GPS] %.1fm from %s (hdop=%.1f)", dist, wp.name, current_gps.hdop)
            if dist < CFG.navigation.waypoint_tolerance_m:
                log.info("[GPS] Within %.1fm - handing to SLAM", dist)
                self.state = NavState.SLAM_LOCALISE

        elif self.state == NavState.SLAM_LOCALISE:
            _pose = self.slam.get_pose()
            if wp.local is None:
                wp.local = Pose2D(0, 0, 0)
            dist = self.slam.distance_to_local(wp.local)
            log.debug("[SLAM] %.2fm from local target", dist)
            if dist < 0.1:
                log.info("[SLAM] Close enough - starting vision alignment")
                self.state = NavState.ALIGNING

        elif self.state == NavState.ALIGNING:
            log.debug("[ALIGN] Waiting for alignment system...")

        elif self.state == NavState.DRILLING:
            log.info("[DRILL] Drilling at %s", wp.name)

        elif self.state == NavState.MEASURING:
            log.info("[MEASURE] Measuring at %s", wp.name)

        return self.state

    def notify_aligned(self):
        if self.state == NavState.ALIGNING:
            self.state = NavState.DRILLING

    def notify_drill_complete(self):
        if self.state == NavState.DRILLING:
            wp = self.current_waypoint
            if wp:
                wp.drilled = True
            self.state = NavState.MEASURING

    def notify_measure_complete(self):
        if self.state == NavState.MEASURING:
            wp = self.current_waypoint
            if wp:
                wp.measured = True
            self.current_wp_idx += 1
            self.state = NavState.GPS_NAVIGATE if self.current_waypoint else NavState.DONE
            log.info(
                "Waypoint complete. Next: %s",
                self.current_waypoint.name if self.current_waypoint else "DONE",
            )

    def run_demo(self, ticks: int = 60):
        log.info("=== Navigator demo: %d waypoints, %d ticks ===", len(self.waypoints), ticks)
        self.state = NavState.IDLE
        tick_delay = 0.2

        for tick in range(ticks):
            state = self.step()
            log.info(
                "[tick %3d] State: %-16s  WP: %s",
                tick,
                state.name,
                self.current_waypoint.name if self.current_waypoint else "—",
            )

            if state == NavState.ALIGNING:
                time.sleep(0.5)
                self.notify_aligned()
            elif state == NavState.DRILLING:
                time.sleep(0.5)
                self.notify_drill_complete()
            elif state == NavState.MEASURING:
                time.sleep(0.3)
                self.notify_measure_complete()
            elif state == NavState.DONE:
                log.info("All waypoints complete!")
                break

            time.sleep(tick_delay)

    def run_mission(self, ticks: int = 120, tick_delay: float = 0.2) -> list[dict]:
        """Drive through the configured waypoint list and return a summary of progress."""
        if not self.waypoints:
            log.warning("No waypoints configured for mission")
            return []

        results: list[dict] = []
        self.state = NavState.IDLE
        self.current_wp_idx = 0

        for tick in range(ticks):
            if self.current_waypoint is None:
                self.state = NavState.DONE
                break

            current = self.current_waypoint
            gps = self.gps.read()
            dist = gps.distance_to(current.gps)
            bearing = gps.bearing_to(current.gps)
            heading_error = (bearing - self.slam.get_pose().yaw) % 360
            if heading_error > 180:
                heading_error -= 360

            log.info(
                "Mission tick %d: %s | dist=%.2fm | bearing=%.1fdeg | heading_error=%.1fdeg | action=%s",
                tick,
                current.name,
                dist,
                bearing,
                heading_error,
                current.action,
            )

            if dist <= current.tolerance_m:
                results.append({
                    "waypoint": current.name,
                    "status": "reached",
                    "distance_m": round(dist, 3),
                    "action": current.action,
                })
                log.info("Reached waypoint %s (%s)", current.name, current.action)
                self.current_wp_idx += 1
                self.state = NavState.GPS_NAVIGATE if self.current_waypoint else NavState.DONE
                if self.state == NavState.DONE:
                    break
            time.sleep(tick_delay)

        log.info("Mission complete: %d waypoints reached", len(results))
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

    waypoints = [
        Waypoint("Sample A", GPSCoord(30.6281, -96.3344)),
        Waypoint("Sample B", GPSCoord(30.6283, -96.3341)),
        Waypoint("Sample C", GPSCoord(30.6285, -96.3338)),
    ]

    nav = Navigator(waypoints)
    nav.run_demo(ticks=80)
