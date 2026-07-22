"""
DIRT Bot - 3D Rover Simulation
===============================

Renders a rover moving through a 3D terrain with a simple chase camera,
waypoint following, and a field log that can be saved to CSV.
        cv2.putText(panel, "speed / pitch / roll", (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
This complements the alignment simulator by showing the rover traveling
through an environment instead of only correcting a camera pose.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger("rover3d")


CONTROL_PROFILES = {
    "smooth": {"max_speed_mps": 0.55, "steering_gain": 1.35, "accel_mps2": 0.28},
    "pursuit": {"max_speed_mps": 0.75, "steering_gain": 2.00, "accel_mps2": 0.45},
    "aggressive": {"max_speed_mps": 1.00, "steering_gain": 2.80, "accel_mps2": 0.62},
}

TERRAIN_PROFILES = {"rolling", "dunes", "rocky"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _rotation_matrix(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


@dataclass
class Waypoint3D:
    name: str
    x: float
    y: float


@dataclass
class RoverState:
    x: float
    y: float
    heading_deg: float
    speed: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    z: float = 0.0


@dataclass
class RoverSample:
    step: int
    run_label: str
    control_mode: str
    terrain_mode: str
    target: str
    x: float
    y: float
    z: float
    heading_deg: float
    speed: float
    target_distance_m: float
    target_bearing_deg: float
    terrain_height: float
    pitch_deg: float
    roll_deg: float
    wheel_slip: float
    power_draw: float
    path_curvature: float
    reached: bool


@dataclass
class MotionLog:
    rows: List[RoverSample] = field(default_factory=list)

    def record(
        self,
        step: int,
        run_label: str,
        control_mode: str,
        terrain_mode: str,
        target: str,
        state: RoverState,
        target_distance_m: float,
        target_bearing_deg: float,
        terrain_height: float,
        wheel_slip: float,
        power_draw: float,
        path_curvature: float,
        reached: bool,
    ) -> None:
        self.rows.append(
            RoverSample(
                step=step,
                run_label=run_label,
                control_mode=control_mode,
                terrain_mode=terrain_mode,
                target=target,
                x=round(state.x, 3),
                y=round(state.y, 3),
                z=round(state.z, 3),
                heading_deg=round(state.heading_deg, 2),
                speed=round(state.speed, 3),
                target_distance_m=round(target_distance_m, 3),
                target_bearing_deg=round(target_bearing_deg, 2),
                terrain_height=round(terrain_height, 3),
                pitch_deg=round(state.pitch_deg, 2),
                roll_deg=round(state.roll_deg, 2),
                wheel_slip=round(wheel_slip, 3),
                power_draw=round(power_draw, 3),
                path_curvature=round(path_curvature, 3),
                reached=reached,
            )
        )

    def save_csv(self, path: str = "sim_rover_3d_path.csv") -> None:
        if not self.rows:
            return
        with open(path, "w", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=list(self.rows[0].__dict__.keys()))
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row.__dict__)
        log.info("3D path log saved -> %s", path)


class TerrainField:
    def __init__(self, span_m: float = 24.0, terrain_mode: str = "rolling"):
        self.span_m = span_m
        self.terrain_mode = terrain_mode if terrain_mode in TERRAIN_PROFILES else "rolling"

    def height(self, x: float, y: float) -> float:
        if self.terrain_mode == "dunes":
            return 0.55 * math.sin(x / 2.4) + 0.25 * math.sin((x + y) / 3.6) + 0.14 * math.cos(y / 4.4)
        if self.terrain_mode == "rocky":
            return (
                0.35 * math.sin(x / 1.8)
                + 0.25 * math.cos(y / 2.2)
                + 0.12 * math.sin((x * 1.7 + y * 1.3) / 1.6)
                + 0.08 * math.cos(math.hypot(x, y) / 1.4)
            )
        return (
            0.45 * math.sin(x / 2.9)
            + 0.35 * math.cos(y / 3.4)
            + 0.18 * math.sin((x + y) / 4.5)
            + 0.08 * math.cos(math.hypot(x, y) / 2.7)
        )

    def gradient(self, x: float, y: float, delta: float = 0.15) -> Tuple[float, float]:
        dx = (self.height(x + delta, y) - self.height(x - delta, y)) / (2.0 * delta)
        dy = (self.height(x, y + delta) - self.height(x, y - delta)) / (2.0 * delta)
        return dx, dy


class Rover3DSim:
    def __init__(
        self,
        waypoints: Optional[Sequence[Waypoint3D]] = None,
        terrain_span_m: float = 24.0,
        waypoint_tolerance_m: float = 0.45,
        max_speed_mps: Optional[float] = None,
        steering_gain: Optional[float] = None,
        accel_mps2: Optional[float] = None,
        control_mode: str = "pursuit",
        terrain_mode: str = "rolling",
    ):
        if control_mode not in CONTROL_PROFILES:
            control_mode = "pursuit"
        if terrain_mode not in TERRAIN_PROFILES:
            terrain_mode = "rolling"

        profile = CONTROL_PROFILES[control_mode]
        max_speed_mps = profile["max_speed_mps"] if max_speed_mps is None else max_speed_mps
        steering_gain = profile["steering_gain"] if steering_gain is None else steering_gain
        accel_mps2 = profile["accel_mps2"] if accel_mps2 is None else accel_mps2

        self.control_mode = control_mode
        self.terrain_mode = terrain_mode
        self.terrain = TerrainField(span_m=terrain_span_m, terrain_mode=terrain_mode)
        self.waypoint_tolerance_m = waypoint_tolerance_m
        self.max_speed_mps = max_speed_mps
        self.steering_gain = steering_gain
        self.accel_mps2 = accel_mps2
        self.waypoints = list(waypoints or self._default_waypoints())
        self.state = RoverState(x=-8.0, y=-7.0, heading_deg=25.0)
        self.target_idx = 0
        self.log = MotionLog()
        self.trail: List[Tuple[float, float, float]] = []
        self.history_speed: List[float] = []
        self.history_pitch: List[float] = []
        self.history_roll: List[float] = []
        self.manual_mode = False
        self.paused = False
        self.manual_throttle = 0.0
        self.manual_steer = 0.0
        self._bounds = terrain_span_m / 2.0
        self.state.z = self.terrain.height(self.state.x, self.state.y)

    def _default_waypoints(self) -> List[Waypoint3D]:
        return [
            Waypoint3D("Start", -8.0, -7.0),
            Waypoint3D("Sample A", -1.5, -2.5),
            Waypoint3D("Sample B", 4.0, 1.5),
            Waypoint3D("Sample C", 8.0, 6.0),
        ]

    def current_waypoint(self) -> Optional[Waypoint3D]:
        if self.target_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.target_idx]

    def _advance_target(self) -> None:
        self.target_idx += 1

    def _limit_to_world(self) -> None:
        self.state.x = _clamp(self.state.x, -self._bounds, self._bounds)
        self.state.y = _clamp(self.state.y, -self._bounds, self._bounds)

    def _update_body_orientation(self) -> None:
        slope_x, slope_y = self.terrain.gradient(self.state.x, self.state.y)
        self.state.pitch_deg = math.degrees(math.atan(-slope_x))
        self.state.roll_deg = math.degrees(math.atan(slope_y))

    def _estimate_path_curvature(self, target: Optional[Waypoint3D], heading_error: float, distance: float) -> float:
        if target is None:
            return 0.0
        if distance < 1e-6:
            return 0.0
        return abs(heading_error) / max(distance, 0.1)

    def _estimate_power_draw(self, heading_error: float, slope_mag: float, speed: float) -> float:
        base = 14.0
        motion_cost = 7.5 * speed
        steering_cost = 0.06 * abs(heading_error)
        slope_cost = 4.0 * slope_mag
        return base + motion_cost + steering_cost + slope_cost

    def _estimate_wheel_slip(self, slope_mag: float, heading_error: float) -> float:
        slip = 0.03 + 0.12 * slope_mag + 0.0025 * abs(heading_error)
        return _clamp(slip, 0.0, 0.45)

    def toggle_manual_mode(self) -> None:
        self.manual_mode = not self.manual_mode

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def reset_pose(self) -> None:
        self.state = RoverState(x=-8.0, y=-7.0, heading_deg=25.0, speed=0.0, z=self.terrain.height(-8.0, -7.0))
        self.target_idx = 0
        self.trail.clear()
        self.history_speed.clear()
        self.history_pitch.clear()
        self.history_roll.clear()
        self.manual_throttle = 0.0
        self.manual_steer = 0.0

    def handle_key(self, key: int) -> bool:
        if key in (-1, 255):
            return False

        key = key & 0xFF
        if key == ord("q"):
            return True
        if key == ord("m"):
            self.toggle_manual_mode()
        elif key == ord(" "):
            self.toggle_pause()
        elif key == ord("r"):
            self.reset_pose()
        elif key in (ord("w"), ord("W")):
            self.manual_throttle = _clamp(self.manual_throttle + 0.15, -1.0, 1.0)
            self.manual_mode = True
        elif key in (ord("s"), ord("S")):
            self.manual_throttle = _clamp(self.manual_throttle - 0.15, -1.0, 1.0)
            self.manual_mode = True
        elif key in (ord("a"), ord("A")):
            self.manual_steer = _clamp(self.manual_steer + 0.15, -1.0, 1.0)
            self.manual_mode = True
        elif key in (ord("d"), ord("D")):
            self.manual_steer = _clamp(self.manual_steer - 0.15, -1.0, 1.0)
            self.manual_mode = True
        elif key in (ord("x"), ord("X")):
            self.manual_throttle = 0.0
            self.manual_steer = 0.0
        elif key in (ord("1"),):
            self.manual_mode = False
        return False

    def _step_manual(self, dt: float) -> None:
        steer_rate = 85.0 * self.manual_steer
        self.state.heading_deg = _wrap_angle_deg(self.state.heading_deg + steer_rate * dt)

        target_speed = self.max_speed_mps * self.manual_throttle
        accel = self.accel_mps2 * 1.4
        if self.state.speed < target_speed:
            self.state.speed = min(target_speed, self.state.speed + accel * dt)
        else:
            self.state.speed = max(target_speed, self.state.speed - accel * dt)

        heading_rad = math.radians(self.state.heading_deg)
        self.state.x += math.cos(heading_rad) * self.state.speed * dt
        self.state.y += math.sin(heading_rad) * self.state.speed * dt
        self._limit_to_world()
        self.state.z = self.terrain.height(self.state.x, self.state.y)
        self._update_body_orientation()
        self.trail.append((self.state.x, self.state.y, self.state.z))
        self.history_speed.append(self.state.speed)
        self.history_pitch.append(self.state.pitch_deg)
        self.history_roll.append(self.state.roll_deg)

    def _step_autonomous(self, dt: float) -> bool:
        target = self.current_waypoint()
        if target is None:
            self.state.speed = 0.0
            self._update_body_orientation()
            self.trail.append((self.state.x, self.state.y, self.state.z))
            return True

        dx = target.x - self.state.x
        dy = target.y - self.state.y
        distance = math.hypot(dx, dy)
        desired_heading = math.degrees(math.atan2(dy, dx))
        heading_error = _wrap_angle_deg(desired_heading - self.state.heading_deg)
        slope_x, slope_y = self.terrain.gradient(self.state.x, self.state.y)
        slope_mag = math.hypot(slope_x, slope_y)

        steering = _clamp(heading_error * self.steering_gain, -90.0, 90.0)
        self.state.heading_deg = _wrap_angle_deg(self.state.heading_deg + steering * dt)

        target_speed = self.max_speed_mps * _clamp(distance / 2.5, 0.25, 1.0)
        if self.state.speed < target_speed:
            self.state.speed = min(target_speed, self.state.speed + self.accel_mps2 * dt)
        else:
            self.state.speed = max(target_speed, self.state.speed - self.accel_mps2 * dt)

        heading_rad = math.radians(self.state.heading_deg)
        self.state.x += math.cos(heading_rad) * self.state.speed * dt
        self.state.y += math.sin(heading_rad) * self.state.speed * dt
        self._limit_to_world()
        self.state.z = self.terrain.height(self.state.x, self.state.y)
        self._update_body_orientation()
        self.trail.append((self.state.x, self.state.y, self.state.z))
        self.history_speed.append(self.state.speed)
        self.history_pitch.append(self.state.pitch_deg)
        self.history_roll.append(self.state.roll_deg)

        if distance <= self.waypoint_tolerance_m:
            log.info("Reached waypoint %s", target.name)
            self._advance_target()
        return self.current_waypoint() is None

    def step(self, dt: float = 0.12) -> bool:
        if self.paused:
            self._update_body_orientation()
            return False
        if self.manual_mode:
            self._step_manual(dt)
            return False
        return self._step_autonomous(dt)

    def _world_to_camera(
        self,
        point: np.ndarray,
        camera_pos: np.ndarray,
        camera_target: np.ndarray,
        camera_up: np.ndarray,
    ) -> Optional[Tuple[float, float, float]]:
        forward = camera_target - camera_pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            return None
        forward = forward / forward_norm
        right = np.cross(forward, camera_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            return None
        right = right / right_norm
        up = np.cross(right, forward)

        rel = point - camera_pos
        x_cam = float(np.dot(rel, right))
        y_cam = float(np.dot(rel, up))
        z_cam = float(np.dot(rel, forward))
        if z_cam <= 0.15:
            return None
        return x_cam, y_cam, z_cam

    def _project(
        self,
        point: Tuple[float, float, float],
        camera_pos: np.ndarray,
        camera_target: np.ndarray,
        camera_up: np.ndarray,
        w: int,
        h: int,
        focal: float,
    ) -> Optional[Tuple[int, int, float]]:
        cam = self._world_to_camera(np.array(point, dtype=float), camera_pos, camera_target, camera_up)
        if cam is None:
            return None
        x_cam, y_cam, z_cam = cam
        x_px = int(w / 2 + focal * (x_cam / z_cam))
        y_px = int(h * 0.66 - focal * (y_cam / z_cam))
        return x_px, y_px, z_cam

    def _draw_line_3d(
        self,
        canvas: np.ndarray,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        colour: Tuple[int, int, int],
        thickness: int,
        camera_pos: np.ndarray,
        camera_target: np.ndarray,
        camera_up: np.ndarray,
        focal: float,
    ) -> None:
        h, w = canvas.shape[:2]
        a = self._project(p1, camera_pos, camera_target, camera_up, w, h, focal)
        b = self._project(p2, camera_pos, camera_target, camera_up, w, h, focal)
        if a is None or b is None:
            return
        cv2.line(canvas, (a[0], a[1]), (b[0], b[1]), colour, thickness, cv2.LINE_AA)

    def _render_minimap(self, size: Tuple[int, int] = (290, 290)) -> np.ndarray:
        w, h = size
        pad = 18
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        panel[:] = (18, 16, 14)
        cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (70, 64, 58), 1)

        xs = np.linspace(-self._bounds, self._bounds, 48)
        ys = np.linspace(-self._bounds, self._bounds, 48)
        hmin = min(self.terrain.height(x, y) for x in xs for y in ys)
        hmax = max(self.terrain.height(x, y) for x in xs for y in ys)
        span = max(hmax - hmin, 1e-6)

        def to_px(x: float, y: float) -> Tuple[int, int]:
            px = int(pad + ((x + self._bounds) / (2 * self._bounds)) * (w - 2 * pad))
            py = int(h - pad - ((y + self._bounds) / (2 * self._bounds)) * (h - 2 * pad))
            return px, py

        for gx in np.linspace(-self._bounds, self._bounds, 11):
            prev = None
            for gy in np.linspace(-self._bounds, self._bounds, 80):
                z = self.terrain.height(gx, gy)
                shade = int(_clamp(55 + ((z - hmin) / span) * 125, 40, 200))
                point = to_px(gx, gy)
                if prev is not None:
                    cv2.line(panel, prev, point, (shade, shade, shade), 1, cv2.LINE_AA)
                prev = point

        for gy in np.linspace(-self._bounds, self._bounds, 11):
            prev = None
            for gx in np.linspace(-self._bounds, self._bounds, 80):
                z = self.terrain.height(gx, gy)
                shade = int(_clamp(65 + ((z - hmin) / span) * 115, 45, 190))
                point = to_px(gx, gy)
                if prev is not None:
                    cv2.line(panel, prev, point, (shade, shade, shade), 1, cv2.LINE_AA)
                prev = point

        for idx in range(1, len(self.trail)):
            cv2.line(panel, to_px(self.trail[idx - 1][0], self.trail[idx - 1][1]), to_px(self.trail[idx][0], self.trail[idx][1]), (0, 150, 255), 2, cv2.LINE_AA)

        for idx, waypoint in enumerate(self.waypoints):
            px, py = to_px(waypoint.x, waypoint.y)
            colour = (0, 255, 180) if idx == self.target_idx else (0, 220, 255)
            cv2.circle(panel, (px, py), 5, colour, -1, cv2.LINE_AA)
            cv2.putText(panel, waypoint.name, (px + 8, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)

        rx, ry = to_px(self.state.x, self.state.y)
        cv2.circle(panel, (rx, ry), 7, (255, 255, 255), -1, cv2.LINE_AA)
        heading_rad = math.radians(self.state.heading_deg)
        hx = int(rx + math.cos(heading_rad) * 18)
        hy = int(ry - math.sin(heading_rad) * 18)
        cv2.arrowedLine(panel, (rx, ry), (hx, hy), (0, 0, 255), 2, cv2.LINE_AA, 0, 0.28)
        cv2.putText(panel, "Minimap", (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        return panel

    def _render_history_panel(self, size: Tuple[int, int] = (290, 290)) -> np.ndarray:
        w, h = size
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        panel[:] = (14, 18, 18)
        cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (70, 64, 58), 1)

        history = max(1, len(self.history_speed))
        speed_series = self.history_speed[-history:]
        pitch_series = self.history_pitch[-history:]
        roll_series = self.history_roll[-history:]

        def normalize(series: List[float], minimum: float, maximum: float) -> List[Tuple[int, int]]:
            if len(series) < 2:
                return []
            points: List[Tuple[int, int]] = []
            for idx, value in enumerate(series):
                x = int(24 + idx / max(len(series) - 1, 1) * (w - 44))
                y = int(h // 2 - ((value - minimum) / max(maximum - minimum, 1e-6) - 0.5) * (h - 54))
                points.append((x, y))
            return points

        cv2.line(panel, (20, h // 2), (w - 20, h // 2), (70, 70, 70), 1, cv2.LINE_AA)
        for pts, color in [
            (normalize(speed_series, 0.0, max(1.2, max(self.history_speed or [1.0]))), (0, 180, 255)),
            (normalize(pitch_series, -20.0, 20.0), (0, 255, 120)),
            (normalize(roll_series, -20.0, 20.0), (255, 170, 60)),
        ]:
            for idx in range(1, len(pts)):
                cv2.line(panel, pts[idx - 1], pts[idx], color, 2, cv2.LINE_AA)

        cv2.putText(panel, "History", (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(panel, "speed / pitch / roll", (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
        return panel

    def _world_point(self, dx: float, dy: float, dz: float = 0.0) -> Tuple[float, float, float]:
        heading = math.radians(self.state.heading_deg)
        c = math.cos(heading)
        s = math.sin(heading)
        x = self.state.x + c * dx - s * dy
        y = self.state.y + s * dx + c * dy
        z = self.terrain.height(x, y) + dz
        return x, y, z

    def render(self, resolution: Tuple[int, int] = (1280, 720)) -> np.ndarray:
        width, height = resolution
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Sky and ground gradient.
        for row in range(height):
            t = row / max(height - 1, 1)
            sky = np.array([98, 146, 205], dtype=np.float32)
            haze = np.array([44, 34, 28], dtype=np.float32)
            colour = sky * (1.0 - t) + haze * t
            canvas[row, :] = np.clip(colour, 0, 255).astype(np.uint8)

        cam_heading = math.radians(self.state.heading_deg)
        camera_pos = np.array(
            [
                self.state.x - math.cos(cam_heading) * 5.5,
                self.state.y - math.sin(cam_heading) * 5.5,
                self.state.z + 3.8,
            ],
            dtype=float,
        )
        camera_target = np.array([self.state.x + 2.0, self.state.y + 1.0, self.state.z + 0.2], dtype=float)
        camera_up = np.array([0.0, 0.0, 1.0], dtype=float)
        focal = 900.0

        grid = np.linspace(-self._bounds, self._bounds, 13)
        for x in grid:
            prev = None
            for y in np.linspace(-self._bounds, self._bounds, 64):
                point = (x, y, self.terrain.height(x, y))
                if prev is not None:
                    height_sample = (prev[2] + point[2]) * 0.5
                    shade = int(_clamp(120 + height_sample * 26 - 6 * abs(x), 55, 185))
                    self._draw_line_3d(canvas, prev, point, (shade, shade, shade), 1, camera_pos, camera_target, camera_up, focal)
                prev = point
        for y in grid:
            prev = None
            for x in np.linspace(-self._bounds, self._bounds, 64):
                point = (x, y, self.terrain.height(x, y))
                if prev is not None:
                    height_sample = (prev[2] + point[2]) * 0.5
                    shade = int(_clamp(135 + height_sample * 22 - 5 * abs(y), 65, 190))
                    self._draw_line_3d(canvas, prev, point, (shade, shade, shade), 1, camera_pos, camera_target, camera_up, focal)
                prev = point

        for idx, waypoint in enumerate(self.waypoints):
            base = (waypoint.x, waypoint.y, self.terrain.height(waypoint.x, waypoint.y))
            tip = (waypoint.x, waypoint.y, base[2] + 1.2)
            colour = (0, 255, 180) if idx == self.target_idx else (0, 220, 255)
            self._draw_line_3d(canvas, base, tip, colour, 3, camera_pos, camera_target, camera_up, focal)
            label_point = (waypoint.x + 0.2, waypoint.y - 0.2, base[2] + 1.35)
            label = self._project(label_point, camera_pos, camera_target, camera_up, width, height, focal)
            if label is not None:
                text = waypoint.name
                text_pos = (label[0] + 8, label[1] - 8)
                cv2.rectangle(canvas, (text_pos[0] - 4, text_pos[1] - 16), (text_pos[0] + 92, text_pos[1] + 6), (15, 15, 15), -1)
                cv2.putText(canvas, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)

        for idx in range(1, len(self.trail)):
            self._draw_line_3d(
                canvas,
                self.trail[idx - 1],
                self.trail[idx],
                (0, 140, 255),
                2,
                camera_pos,
                camera_target,
                camera_up,
                focal,
            )

        body_length = 1.5
        body_width = 1.0
        body_height = 0.45
        body_z = self.state.z + 0.18
        body_points = [
            (-body_length / 2, -body_width / 2, 0.0),
            (body_length / 2, -body_width / 2, 0.0),
            (body_length / 2, body_width / 2, 0.0),
            (-body_length / 2, body_width / 2, 0.0),
            (-body_length / 2, -body_width / 2, body_height),
            (body_length / 2, -body_width / 2, body_height),
            (body_length / 2, body_width / 2, body_height),
            (-body_length / 2, body_width / 2, body_height),
        ]
        rot = _rotation_matrix(self.state.heading_deg)
        body_world = []
        for px, py, pz in body_points:
            rotated = rot @ np.array([px, py, 0.0], dtype=float)
            body_world.append((self.state.x + rotated[0], self.state.y + rotated[1], body_z + pz))

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in edges:
            self._draw_line_3d(canvas, body_world[a], body_world[b], (40, 40, 40), 3, camera_pos, camera_target, camera_up, focal)

        # Simple shadow and a front marker so the rover reads more clearly.
        shadow_center = (self.state.x + 0.18, self.state.y - 0.12, self.state.z + 0.01)
        shadow = self._project(shadow_center, camera_pos, camera_target, camera_up, width, height, focal)
        if shadow is not None:
            cv2.ellipse(canvas, (shadow[0], shadow[1] + 6), (28, 14), 0, 0, 360, (16, 16, 16), -1, cv2.LINE_AA)

        self._draw_line_3d(
            canvas,
            (self.state.x, self.state.y, self.state.z + 0.2),
            self._world_point(1.8, 0.0, 0.2),
            (0, 0, 255),
            4,
            camera_pos,
            camera_target,
            camera_up,
            focal,
        )
        self._draw_line_3d(
            canvas,
            (self.state.x, self.state.y, self.state.z + 0.18),
            (self.state.x, self.state.y, self.state.z + 1.3),
            (255, 255, 255),
            2,
            camera_pos,
            camera_target,
            camera_up,
            focal,
        )

        target = self.current_waypoint()
        route = None
        if target is not None:
            route = self._project((target.x, target.y, self.terrain.height(target.x, target.y) + 0.1), camera_pos, camera_target, camera_up, width, height, focal)
        rover_center = self._project((self.state.x, self.state.y, self.state.z + 0.1), camera_pos, camera_target, camera_up, width, height, focal)
        if route is not None and rover_center is not None:
            cv2.line(canvas, (rover_center[0], rover_center[1]), (route[0], route[1]), (0, 180, 255), 2, cv2.LINE_AA)

        minimap = self._render_minimap()
        history_panel = self._render_history_panel()
        canvas[14:304, width - 304:width - 14] = minimap
        canvas[320:610, width - 304:width - 14] = history_panel

        hud_lines = [
            f"Target: {target.name if target else 'none'}",
            f"Pose: x={self.state.x:+.2f} m  y={self.state.y:+.2f} m  z={self.state.z:+.2f} m",
            f"Heading: {self.state.heading_deg:+.1f} deg  Speed: {self.state.speed:.2f} m/s",
            f"Mode: {'manual' if self.manual_mode else self.control_mode}  Terrain: {self.terrain_mode}",
            f"pitch {self.state.pitch_deg:+.1f} deg  roll {self.state.roll_deg:+.1f} deg  throttle {self.manual_throttle:+.2f}  steer {self.manual_steer:+.2f}",
        ]
        for idx, line in enumerate(hud_lines):
            y = 28 + idx * 24
            cv2.rectangle(canvas, (14, y - 18), (14 + 540, y + 4), (18, 18, 18), -1)
            cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

        help_lines = [
            "W/S throttle  A/D steer  M toggle manual  Space pause  R reset  X zero input  1 auto  Q quit",
            "Manual mode drives the rover directly; auto mode follows waypoints.",
        ]
        for idx, line in enumerate(help_lines):
            cv2.putText(canvas, line, (18, height - 72 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)

        cv2.putText(canvas, "DIRT Rover 3D Simulation", (18, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 220), 2, cv2.LINE_AA)
        return canvas

    def save_path_csv(self, path: str = "sim_rover_3d_path.csv") -> None:
        self.log.save_csv(path)


def run_rover_3d_demo(
    waypoints: Optional[Sequence[Waypoint3D]] = None,
    show_video: bool = True,
    save_csv: bool = True,
    delay_ms: int = 40,
    max_steps: int = 420,
    waypoint_tolerance_m: float = 0.45,
    control_mode: str = "pursuit",
    terrain_mode: str = "rolling",
    csv_path: str = "sim_rover_3d_path.csv",
    run_label: str = "demo",
    start_manual: bool = False,
) -> MotionLog:
    sim = Rover3DSim(
        waypoints=waypoints,
        waypoint_tolerance_m=waypoint_tolerance_m,
        control_mode=control_mode,
        terrain_mode=terrain_mode,
    )
    sim.manual_mode = start_manual
    log.info("Starting 3D rover demo with %d waypoints", len(sim.waypoints))

    finished = False
    for step in range(max_steps):
        target = sim.current_waypoint()
        target_name = target.name if target else "done"
        dx = (target.x - sim.state.x) if target else 0.0
        dy = (target.y - sim.state.y) if target else 0.0
        target_distance_m = math.hypot(dx, dy) if target else 0.0
        target_bearing_deg = _wrap_angle_deg(math.degrees(math.atan2(dy, dx)) - sim.state.heading_deg) if target else 0.0
        slope_x, slope_y = sim.terrain.gradient(sim.state.x, sim.state.y)
        slope_mag = math.hypot(slope_x, slope_y)
        path_curvature = sim._estimate_path_curvature(target, target_bearing_deg, target_distance_m)
        power_draw = sim._estimate_power_draw(target_bearing_deg, slope_mag, sim.state.speed)
        wheel_slip = sim._estimate_wheel_slip(slope_mag, target_bearing_deg)

        finished = sim.step()
        terrain_height = sim.terrain.height(sim.state.x, sim.state.y)
        sim.log.record(
            step,
            run_label,
            sim.control_mode,
            sim.terrain_mode,
            target_name,
            sim.state,
            target_distance_m,
            target_bearing_deg,
            terrain_height,
            wheel_slip,
            power_draw,
            path_curvature,
            target is None,
        )

        if show_video:
            frame = sim.render()
            cv2.imshow("DIRT Sim - Rover 3D", frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if sim.handle_key(key):
                log.info("Quit by user")
                break

        if finished:
            log.info("3D rover mission complete in %d steps", step + 1)
            break

    if show_video:
        cv2.destroyAllWindows()

    if save_csv:
        sim.save_path_csv(csv_path)

    if sim.log.rows:
        last = sim.log.rows[-1]
        print("\nRover 3D Simulation Summary")
        print(f"  Steps            : {len(sim.log.rows)}")
        print(f"  Final position   : x={last.x:+.2f} m  y={last.y:+.2f} m")
        print(f"  Final heading    : {last.heading_deg:+.1f} deg")
        print(f"  Final speed      : {last.speed:.2f} m/s")
        print(f"  Target complete  : {finished}")
        print()

    return sim.log


def _build_waypoints_from_csv(path: str) -> List[Waypoint3D]:
    waypoints: List[Waypoint3D] = []
    with open(path, newline="") as file_handle:
        for row in csv.DictReader(file_handle):
            waypoints.append(Waypoint3D(name=row["name"], x=float(row["x"]), y=float(row["y"])))
    return waypoints


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

    parser = argparse.ArgumentParser(description="DIRT rover 3D simulation")
    parser.add_argument("--no-gui", action="store_true", help="Run headless")
    parser.add_argument("--iters", type=int, default=420, help="Max simulation steps")
    parser.add_argument("--speed", type=int, default=40, help="Frame delay ms")
    parser.add_argument("--waypoints", type=str, default=None, help="Optional CSV with columns name,x,y")
    parser.add_argument("--csv", type=str, default="sim_rover_3d_path.csv", help="CSV output path")
    parser.add_argument("--control-mode", choices=sorted(CONTROL_PROFILES.keys()), default="pursuit", help="Motion control profile")
    parser.add_argument("--terrain-mode", choices=sorted(TERRAIN_PROFILES), default="rolling", help="Terrain profile")
    parser.add_argument("--label", type=str, default="demo", help="Run label written into the CSV")
    parser.add_argument("--manual", action="store_true", help="Start in manual driving mode")
    args = parser.parse_args()

    custom_waypoints = _build_waypoints_from_csv(args.waypoints) if args.waypoints else None
    log.info("Controls: W/S throttle, A/D steer, M manual toggle, Space pause, R reset, X zero, 1 auto, Q quit")
    run_rover_3d_demo(
        waypoints=custom_waypoints,
        show_video=not args.no_gui,
        save_csv=True,
        delay_ms=args.speed,
        max_steps=args.iters,
        control_mode=args.control_mode,
        terrain_mode=args.terrain_mode,
        csv_path=args.csv,
        run_label=args.label,
        start_manual=args.manual,
    )