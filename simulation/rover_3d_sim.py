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
import time
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


def _lerp(value: float, target: float, alpha: float) -> float:
    return value + (target - value) * _clamp(alpha, 0.0, 1.0)


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
        self._background_cache: dict[Tuple[int, int], np.ndarray] = {}
        self._terrain_fill_cache: dict[Tuple[str, float], List[Tuple[Tuple[Tuple[float, float, float], ...], Tuple[int, int, int]]]] = {}
        self._terrain_cache: dict[Tuple[str, float], List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[int, int, int]]]] = {}
        self._minimap_base_cache: dict[Tuple[int, int], np.ndarray] = {}
        self._history_base_cache: dict[Tuple[int, int], np.ndarray] = {}
        self._camera_pos = np.zeros(3, dtype=float)
        self._camera_target = np.zeros(3, dtype=float)
        self._render_frame = 0
        self._wheel_rotation = 0.0
        self._dust_particles: List[Tuple[float, float, float, float, float]] = []
        self.state.z = self.terrain.height(self.state.x, self.state.y)
        self._camera_pos = np.array([self.state.x - 5.5, self.state.y - 5.5, self.state.z + 3.8], dtype=float)
        self._camera_target = np.array([self.state.x + 2.0, self.state.y + 1.0, self.state.z + 0.2], dtype=float)

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
        self._dust_particles.clear()
        self._wheel_rotation = 0.0
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

    def _spawn_dust(self) -> None:
        if abs(self.state.speed) < 0.08:
            return
        heading_rad = math.radians(self.state.heading_deg)
        trail_x = self.state.x - math.cos(heading_rad) * 0.7
        trail_y = self.state.y - math.sin(heading_rad) * 0.7
        for _ in range(1 + int(abs(self.state.speed) * 2.5)):
            offset = np.array([np.random.normal(0.0, 0.22), np.random.normal(0.0, 0.22), 0.0])
            self._dust_particles.append((trail_x + offset[0], trail_y + offset[1], self.state.z + 0.04, 0.08 + np.random.random() * 0.18, 0.35 + np.random.random() * 0.65))
        if len(self._dust_particles) > 220:
            self._dust_particles = self._dust_particles[-220:]

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
        self._wheel_rotation += self.state.speed * dt * 4.4
        self.trail.append((self.state.x, self.state.y, self.state.z))
        self.history_speed.append(self.state.speed)
        self.history_pitch.append(self.state.pitch_deg)
        self.history_roll.append(self.state.roll_deg)
        self._spawn_dust()

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
        self._wheel_rotation += self.state.speed * dt * 4.4
        self.trail.append((self.state.x, self.state.y, self.state.z))
        self.history_speed.append(self.state.speed)
        self.history_pitch.append(self.state.pitch_deg)
        self.history_roll.append(self.state.roll_deg)
        self._spawn_dust()

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

    def _fill_polygon_3d(
        self,
        canvas: np.ndarray,
        points: Sequence[Tuple[float, float, float]],
        colour: Tuple[int, int, int],
        camera_pos: np.ndarray,
        camera_target: np.ndarray,
        camera_up: np.ndarray,
        focal: float,
    ) -> None:
        h, w = canvas.shape[:2]
        projected: List[Tuple[int, int]] = []
        for point in points:
            result = self._project(point, camera_pos, camera_target, camera_up, w, h, focal)
            if result is None:
                return
            projected.append((result[0], result[1]))
        cv2.fillConvexPoly(canvas, np.array(projected, dtype=np.int32), colour, cv2.LINE_AA)

    def _projected_radius(
        self,
        point: Tuple[float, float, float],
        offset: Tuple[float, float, float],
        camera_pos: np.ndarray,
        camera_target: np.ndarray,
        camera_up: np.ndarray,
        focal: float,
        width: int,
        height: int,
    ) -> Optional[int]:
        base = self._project(point, camera_pos, camera_target, camera_up, width, height, focal)
        other = self._project((point[0] + offset[0], point[1] + offset[1], point[2] + offset[2]), camera_pos, camera_target, camera_up, width, height, focal)
        if base is None or other is None:
            return None
        radius = int(max(2.0, min(22.0, math.hypot(other[0] - base[0], other[1] - base[1]))))
        return radius

    def _to_world_frame(
        self,
        rot: np.ndarray,
        local_point: Tuple[float, float, float],
        base_z: float,
    ) -> Tuple[float, float, float]:
        rotated = rot @ np.array([local_point[0], local_point[1], 0.0], dtype=float)
        return (self.state.x + rotated[0], self.state.y + rotated[1], base_z + local_point[2])

    def _render_minimap(self, size: Tuple[int, int] = (290, 290)) -> np.ndarray:
        w, h = size
        pad = 18
        panel = self._get_minimap_base(size)

        def to_px(x: float, y: float) -> Tuple[int, int]:
            px = int(pad + ((x + self._bounds) / (2 * self._bounds)) * (w - 2 * pad))
            py = int(h - pad - ((y + self._bounds) / (2 * self._bounds)) * (h - 2 * pad))
            return px, py

        for idx in range(1, len(self.trail)):
            start = to_px(self.trail[idx - 1][0], self.trail[idx - 1][1])
            end = to_px(self.trail[idx][0], self.trail[idx][1])
            cv2.line(panel, start, end, (0, 150, 255), 2, cv2.LINE_AA)

        for idx, waypoint in enumerate(self.waypoints):
            px, py = to_px(waypoint.x, waypoint.y)
            colour = (0, 255, 180) if idx == self.target_idx else (0, 220, 255)
            cv2.circle(panel, (px, py), 5, colour, -1, cv2.LINE_AA)
            cv2.circle(panel, (px, py), 8 if idx == self.target_idx else 7, colour, 1, cv2.LINE_AA)
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
        panel = self._get_history_base(size)

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

    def _build_background(self, resolution: Tuple[int, int]) -> np.ndarray:
        width, height = resolution
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        sky_top = np.array([110, 156, 214], dtype=np.float32)
        sky_bottom = np.array([44, 36, 30], dtype=np.float32)
        glow = np.array([22, 18, 8], dtype=np.float32)
        center_x = width * 0.48
        center_y = height * 0.56

        for row in range(height):
            t = row / max(height - 1, 1)
            colour = sky_top * (1.0 - t) + sky_bottom * t
            if t > 0.48:
                colour = colour + glow * math.exp(-((t - 0.64) ** 2) * 18.0)
            canvas[row, :] = np.clip(colour, 0, 255).astype(np.uint8)

        yy, xx = np.ogrid[:height, :width]
        dx = (xx - center_x) / max(width, 1)
        dy = (yy - center_y) / max(height, 1)
        vignette = np.clip(1.0 - 0.28 * (dx * dx + dy * dy), 0.78, 1.0)
        canvas[:] = np.clip(canvas.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

        return canvas

    def _get_background(self, resolution: Tuple[int, int]) -> np.ndarray:
        key = (int(resolution[0]), int(resolution[1]))
        if key not in self._background_cache:
            self._background_cache[key] = self._build_background(key)
        return self._background_cache[key].copy()

    def _get_terrain_segments(self) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[int, int, int]]]:
        key = (self.terrain_mode, self._bounds)
        if key in self._terrain_cache:
            return self._terrain_cache[key]

        segments: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[int, int, int]]] = []
        grid = np.linspace(-self._bounds, self._bounds, 13)
        samples = np.linspace(-self._bounds, self._bounds, 64)

        for x in grid:
            prev = None
            for y in samples:
                point = (float(x), float(y), self.terrain.height(float(x), float(y)))
                if prev is not None:
                    height_sample = (prev[2] + point[2]) * 0.5
                    shade = int(_clamp(120 + height_sample * 26 - 6 * abs(float(x)), 55, 185))
                    segments.append((prev, point, (shade, shade, shade)))
                prev = point

        for y in grid:
            prev = None
            for x in samples:
                point = (float(x), float(y), self.terrain.height(float(x), float(y)))
                if prev is not None:
                    height_sample = (prev[2] + point[2]) * 0.5
                    shade = int(_clamp(135 + height_sample * 22 - 5 * abs(float(y)), 65, 190))
                    segments.append((prev, point, (shade, shade, shade)))
                prev = point

        self._terrain_cache[key] = segments
        return segments

    def _get_terrain_faces(self) -> List[Tuple[Tuple[Tuple[float, float, float], ...], Tuple[int, int, int]]]:
        key = (self.terrain_mode, self._bounds)
        if key in self._terrain_fill_cache:
            return self._terrain_fill_cache[key]

        faces: List[Tuple[Tuple[Tuple[float, float, float], ...], Tuple[int, int, int]]] = []
        grid = np.linspace(-self._bounds, self._bounds, 12)

        for ix in range(len(grid) - 1):
            x0 = float(grid[ix])
            x1 = float(grid[ix + 1])
            for iy in range(len(grid) - 1):
                y0 = float(grid[iy])
                y1 = float(grid[iy + 1])
                p00 = (x0, y0, self.terrain.height(x0, y0))
                p10 = (x1, y0, self.terrain.height(x1, y0))
                p11 = (x1, y1, self.terrain.height(x1, y1))
                p01 = (x0, y1, self.terrain.height(x0, y1))
                avg_height = (p00[2] + p10[2] + p11[2] + p01[2]) * 0.25
                slope_x, slope_y = self.terrain.gradient((x0 + x1) * 0.5, (y0 + y1) * 0.5)
                slope_mag = min(1.8, math.hypot(slope_x, slope_y))
                warm = np.array([92, 76, 54], dtype=float)
                cool = np.array([62, 102, 88], dtype=float)
                blend = _clamp((avg_height + 0.8) / 1.8, 0.0, 1.0)
                base = warm * (1.0 - blend) + cool * blend
                shade = _clamp(1.0 - 0.24 * slope_mag - 0.12 * max(0.0, avg_height), 0.52, 1.0)
                colour = (
                    int(_clamp(base[0] * shade, 0, 255)),
                    int(_clamp(base[1] * shade, 0, 255)),
                    int(_clamp(base[2] * shade, 0, 255)),
                )
                faces.append(((p00, p10, p11, p01), colour))

        self._terrain_fill_cache[key] = faces
        return faces

    def _get_minimap_base(self, size: Tuple[int, int] = (290, 290)) -> np.ndarray:
        key = (int(size[0]), int(size[1]))
        if key in self._minimap_base_cache:
            return self._minimap_base_cache[key].copy()

        w, h = key
        pad = 18
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        panel[:] = (18, 16, 14)
        cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (78, 70, 62), 1)
        cv2.putText(panel, "MINIMAP", (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)

        xs = np.linspace(-self._bounds, self._bounds, 48)
        ys = np.linspace(-self._bounds, self._bounds, 48)
        hmin = min(self.terrain.height(float(x), float(y)) for x in xs for y in ys)
        hmax = max(self.terrain.height(float(x), float(y)) for x in xs for y in ys)
        span = max(hmax - hmin, 1e-6)

        def to_px(x: float, y: float) -> Tuple[int, int]:
            px = int(pad + (x + self._bounds) / (2 * self._bounds) * (w - 2 * pad))
            py = int(h - pad - (y + self._bounds) / (2 * self._bounds) * (h - 2 * pad))
            return px, py

        for x in xs:
            prev = None
            for y in ys:
                z = self.terrain.height(float(x), float(y))
                shade = int(_clamp(62 + ((z - hmin) / span) * 110, 42, 188))
                point = to_px(float(x), float(y))
                if prev is not None:
                    cv2.line(panel, prev, point, (shade, shade, shade), 1, cv2.LINE_AA)
                prev = point

        for y in ys:
            prev = None
            for x in xs:
                z = self.terrain.height(float(x), float(y))
                shade = int(_clamp(62 + ((z - hmin) / span) * 110, 42, 188))
                point = to_px(float(x), float(y))
                if prev is not None:
                    cv2.line(panel, prev, point, (shade, shade, shade), 1, cv2.LINE_AA)
                prev = point

        cv2.rectangle(panel, (pad - 2, pad - 2), (w - pad + 2, h - pad + 2), (90, 80, 70), 1)
        self._minimap_base_cache[key] = panel
        return panel.copy()

    def _get_history_base(self, size: Tuple[int, int] = (290, 290)) -> np.ndarray:
        key = (int(size[0]), int(size[1]))
        if key in self._history_base_cache:
            return self._history_base_cache[key].copy()

        w, h = key
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        panel[:] = (14, 18, 18)
        cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (78, 70, 62), 1)
        cv2.putText(panel, "HISTORY", (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.line(panel, (20, h // 2), (w - 20, h // 2), (75, 75, 75), 1, cv2.LINE_AA)
        cv2.line(panel, (20, 42), (20, h - 28), (68, 68, 68), 1, cv2.LINE_AA)
        for y in [64, 120, 176, 232]:
            cv2.line(panel, (22, y), (w - 22, y), (32, 38, 38), 1, cv2.LINE_AA)
        self._history_base_cache[key] = panel
        return panel.copy()

    def _render_dust(self, canvas: np.ndarray, camera_pos: np.ndarray, camera_target: np.ndarray, camera_up: np.ndarray, focal: float, width: int, height: int) -> None:
        for x, y, z, radius, alpha in self._dust_particles:
            dust_point = (x, y, z)
            p = self._project(dust_point, camera_pos, camera_target, camera_up, width, height, focal)
            if p is None:
                continue
            px, py = p[0], p[1]
            intensity = int(_clamp(alpha * 255.0, 20, 180))
            cv2.circle(canvas, (px, py), max(2, int(radius * 8)), (intensity, intensity, intensity), -1, cv2.LINE_AA)

    def render(self, resolution: Tuple[int, int] = (2560, 1440)) -> np.ndarray:
        width, height = resolution
        target_pixels = 900_000
        render_scale = 1.0
        if width * height > target_pixels:
            render_scale = math.sqrt(target_pixels / (width * height))
            width = max(320, int(width * render_scale))
            height = max(180, int(height * render_scale))
        canvas = self._get_background((width, height))

        cam_heading = math.radians(self.state.heading_deg)
        desired_distance = 5.2 + min(2.0, max(0.0, self.state.speed * 0.75))
        desired_eye = np.array(
            [
                self.state.x - math.cos(cam_heading) * desired_distance,
                self.state.y - math.sin(cam_heading) * desired_distance,
                self.state.z + 3.2 + min(1.4, max(0.0, self.state.speed * 0.2)),
            ],
            dtype=float,
        )
        desired_target = np.array(
            [
                self.state.x + math.cos(cam_heading) * 1.8,
                self.state.y + math.sin(cam_heading) * 1.8,
                self.state.z + 0.25 + 0.10 * math.sin(math.radians(self.state.pitch_deg)),
            ],
            dtype=float,
        )

        smooth_alpha = 0.18 if self._render_frame == 0 else 0.12
        self._camera_pos = np.array(
            [
                _lerp(self._camera_pos[0], desired_eye[0], smooth_alpha),
                _lerp(self._camera_pos[1], desired_eye[1], smooth_alpha),
                _lerp(self._camera_pos[2], desired_eye[2], smooth_alpha),
            ],
            dtype=float,
        )
        self._camera_target = np.array(
            [
                _lerp(self._camera_target[0], desired_target[0], smooth_alpha),
                _lerp(self._camera_target[1], desired_target[1], smooth_alpha),
                _lerp(self._camera_target[2], desired_target[2], smooth_alpha),
            ],
            dtype=float,
        )
        self._render_frame += 1

        roll_tilt = math.radians(self.state.roll_deg) * 0.35
        camera_pos = self._camera_pos.copy()
        camera_target = self._camera_target.copy()
        camera_up = np.array([math.sin(roll_tilt) * 0.35, -math.sin(roll_tilt) * 0.18, 1.0], dtype=float)
        camera_up /= np.linalg.norm(camera_up)
        focal = 980.0 * render_scale

        for face, colour in self._get_terrain_faces():
            projected: List[Tuple[int, int]] = []
            visible = True
            for point in face:
                projected_point = self._project(point, camera_pos, camera_target, camera_up, width, height, focal)
                if projected_point is None:
                    visible = False
                    break
                projected.append((projected_point[0], projected_point[1]))
            if visible and len(projected) == 4:
                cv2.fillConvexPoly(canvas, np.array(projected, dtype=np.int32), colour)

        for prev, point, colour in self._get_terrain_segments():
            muted = (int(colour[0] * 0.78), int(colour[1] * 0.78), int(colour[2] * 0.78))
            self._draw_line_3d(canvas, prev, point, muted, 1, camera_pos, camera_target, camera_up, focal)

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

        body_length = 1.9
        body_width = 1.15
        body_height = 0.42
        cabin_height = 0.7
        body_z = self.state.z + 0.18
        body_points = [
            (-body_length / 2, -body_width / 2, 0.0),
            (body_length / 2, -body_width / 2, 0.0),
            (body_length / 2, body_width / 2, 0.0),
            (-body_length / 2, body_width / 2, 0.0),
            (-body_length / 2 + 0.22, -body_width / 2 + 0.08, body_height),
            (body_length / 2 - 0.18, -body_width / 2 + 0.08, body_height),
            (body_length / 2 - 0.18, body_width / 2 - 0.08, body_height),
            (-body_length / 2 + 0.22, body_width / 2 - 0.08, body_height),
        ]
        rot = _rotation_matrix(self.state.heading_deg)
        body_world = [self._to_world_frame(rot, point, body_z) for point in body_points]

        cabin_points = [
            (-0.42, -0.36, body_height + 0.02),
            (0.62, -0.36, body_height + 0.02),
            (0.62, 0.36, body_height + 0.02),
            (-0.42, 0.36, body_height + 0.02),
            (-0.28, -0.30, cabin_height),
            (0.49, -0.30, cabin_height),
            (0.49, 0.30, cabin_height),
            (-0.28, 0.30, cabin_height),
        ]
        cabin_world = [self._to_world_frame(rot, point, body_z) for point in cabin_points]

        hood_points = [
            (0.58, -0.34, body_height * 0.92),
            (0.98, -0.28, body_height * 0.82),
            (0.98, 0.28, body_height * 0.82),
            (0.58, 0.34, body_height * 0.92),
        ]
        hood_world = [self._to_world_frame(rot, point, body_z) for point in hood_points]

        roof_points = [
            (-0.12, -0.24, cabin_height + 0.06),
            (0.42, -0.24, cabin_height + 0.06),
            (0.42, 0.24, cabin_height + 0.06),
            (-0.12, 0.24, cabin_height + 0.06),
        ]
        roof_world = [self._to_world_frame(rot, point, body_z) for point in roof_points]

        front_bumper = [
            (0.96, -0.38, 0.10),
            (1.12, -0.28, 0.12),
            (1.12, 0.28, 0.12),
            (0.96, 0.38, 0.10),
        ]
        front_bumper_world = [self._to_world_frame(rot, point, body_z) for point in front_bumper]

        rear_panel = [
            (-1.02, -0.32, 0.12),
            (-1.12, -0.26, 0.32),
            (-1.12, 0.26, 0.32),
            (-1.02, 0.32, 0.12),
        ]
        rear_panel_world = [self._to_world_frame(rot, point, body_z) for point in rear_panel]

        self._fill_polygon_3d(canvas, body_world[:4], (44, 45, 48), camera_pos, camera_target, camera_up, focal)
        self._fill_polygon_3d(canvas, body_world[4:], (64, 68, 72), camera_pos, camera_target, camera_up, focal)
        self._fill_polygon_3d(canvas, rear_panel_world, (52, 56, 62), camera_pos, camera_target, camera_up, focal)
        self._fill_polygon_3d(canvas, front_bumper_world, (76, 82, 88), camera_pos, camera_target, camera_up, focal)
        self._fill_polygon_3d(canvas, cabin_world[:4], (92, 105, 116), camera_pos, camera_target, camera_up, focal)
        self._fill_polygon_3d(canvas, cabin_world[4:], (118, 138, 154), camera_pos, camera_target, camera_up, focal)
        self._fill_polygon_3d(canvas, hood_world, (82, 88, 94), camera_pos, camera_target, camera_up, focal)
        self._fill_polygon_3d(canvas, roof_world, (128, 146, 160), camera_pos, camera_target, camera_up, focal)

        highlight_poly = [
            self._to_world_frame(rot, (-0.10, -0.18, body_height + 0.02), body_z),
            self._to_world_frame(rot, (0.45, -0.18, body_height + 0.02), body_z),
            self._to_world_frame(rot, (0.45, -0.03, cabin_height - 0.05), body_z),
            self._to_world_frame(rot, (-0.10, -0.03, cabin_height - 0.05), body_z),
        ]
        self._fill_polygon_3d(canvas, highlight_poly, (156, 168, 180), camera_pos, camera_target, camera_up, focal)

        chamber_points = [
            (-0.68, -0.22, body_height + 0.06),
            (-0.05, -0.22, body_height + 0.06),
            (-0.05, 0.22, body_height + 0.06),
            (-0.68, 0.22, body_height + 0.06),
            (-0.62, -0.16, body_height + 0.42),
            (0.02, -0.16, body_height + 0.42),
            (0.02, 0.16, body_height + 0.42),
            (-0.62, 0.16, body_height + 0.42),
        ]
        chamber_world = [self._to_world_frame(rot, point, body_z) for point in chamber_points]
        self._fill_polygon_3d(canvas, chamber_world[:4], (54, 58, 66), camera_pos, camera_target, camera_up, focal)
        self._fill_polygon_3d(canvas, chamber_world[4:], (86, 146, 166), camera_pos, camera_target, camera_up, focal)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            self._draw_line_3d(canvas, chamber_world[a], chamber_world[b], (170, 212, 220), 1, camera_pos, camera_target, camera_up, focal)

        chamber_window = [
            (-0.52, -0.12, body_height + 0.22),
            (-0.16, -0.12, body_height + 0.22),
            (-0.16, 0.12, body_height + 0.22),
            (-0.52, 0.12, body_height + 0.22),
        ]
        self._fill_polygon_3d(canvas, [self._to_world_frame(rot, point, body_z) for point in chamber_window], (130, 210, 230), camera_pos, camera_target, camera_up, focal)

        scoop_points = [
            (0.84, -0.22, -0.18),
            (1.22, -0.15, -0.28),
            (1.34, 0.00, -0.42),
            (1.22, 0.15, -0.28),
            (0.84, 0.22, -0.18),
            (0.92, 0.00, -0.10),
        ]
        scoop_world = [self._to_world_frame(rot, point, body_z) for point in scoop_points]
        self._fill_polygon_3d(canvas, scoop_world, (120, 98, 58), camera_pos, camera_target, camera_up, focal)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]:
            self._draw_line_3d(canvas, scoop_world[a], scoop_world[b], (202, 176, 96), 2, camera_pos, camera_target, camera_up, focal)

        drill_base = self._to_world_frame(rot, (1.02, 0.0, 0.32), body_z)
        drill_tip = self._to_world_frame(rot, (1.18, 0.0, -0.56), body_z)
        self._draw_line_3d(canvas, drill_base, drill_tip, (238, 210, 84), 4, camera_pos, camera_target, camera_up, focal)
        for offset_y in [-0.08, 0.0, 0.08]:
            auger_start = self._to_world_frame(rot, (0.98, offset_y, 0.18), body_z)
            auger_end = self._to_world_frame(rot, (1.15, offset_y * 0.2, -0.46), body_z)
            self._draw_line_3d(canvas, auger_start, auger_end, (255, 180, 0), 1, camera_pos, camera_target, camera_up, focal)
        drill_bit = self._project(drill_tip, camera_pos, camera_target, camera_up, width, height, focal)
        if drill_bit is not None:
            cv2.circle(canvas, (drill_bit[0], drill_bit[1]), 5, (255, 180, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (drill_bit[0], drill_bit[1]), 10, (255, 220, 120), 1, cv2.LINE_AA)

        sample_chute = [
            (0.74, -0.10, 0.22),
            (0.74, 0.10, 0.22),
            (1.00, 0.10, -0.10),
            (1.00, -0.10, -0.10),
        ]
        self._fill_polygon_3d(canvas, [self._to_world_frame(rot, point, body_z) for point in sample_chute], (102, 88, 52), camera_pos, camera_target, camera_up, focal)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            self._draw_line_3d(canvas, self._to_world_frame(rot, sample_chute[a], body_z), self._to_world_frame(rot, sample_chute[b], body_z), (175, 148, 72), 2, camera_pos, camera_target, camera_up, focal)

        intake_beam_start = self._to_world_frame(rot, (0.96, 0.0, 0.10), body_z)
        intake_beam_end = self._to_world_frame(rot, (-0.18, 0.0, 0.34), body_z)
        self._draw_line_3d(canvas, intake_beam_start, intake_beam_end, (255, 214, 76), 2, camera_pos, camera_target, camera_up, focal)

        wheel_offsets = [
            (-0.78, -0.56, -0.16),
            (-0.78, 0.56, -0.16),
            (0.78, -0.56, -0.16),
            (0.78, 0.56, -0.16),
        ]
        wheel_shadow_offset = (0.16, -0.08, 0.0)
        wheel_span = 0.29
        for wheel_idx, (wx, wy, wz) in enumerate(wheel_offsets):
            offset = rot @ np.array([wx, wy, 0.0], dtype=float)
            suspension_heave = 0.04 * abs(self.state.speed) / max(self.max_speed_mps, 0.1)
            suspension_roll = 0.02 * abs(self.state.roll_deg) / 18.0
            suspension_pitch = 0.02 * abs(self.state.pitch_deg) / 18.0
            wheel_z = body_z + wz - suspension_heave - suspension_roll - suspension_pitch * ((1 if wheel_idx % 2 == 0 else -1))
            wheel_center = (self.state.x + offset[0], self.state.y + offset[1], wheel_z)
            radius = self._projected_radius(wheel_center, (wheel_span, 0.0, 0.0), camera_pos, camera_target, camera_up, focal, width, height)
            projected_point = self._project(wheel_center, camera_pos, camera_target, camera_up, width, height, focal)
            if projected_point is None or radius is None:
                continue
            projected_x, projected_y = projected_point[0], projected_point[1]

            wheel_shadow = self._project((wheel_center[0] + wheel_shadow_offset[0], wheel_center[1] + wheel_shadow_offset[1], wheel_center[2] - 0.03), camera_pos, camera_target, camera_up, width, height, focal)
            if wheel_shadow is not None:
                cv2.ellipse(
                    canvas,
                    (wheel_shadow[0], wheel_shadow[1] + 5),
                    (max(8, radius + 6), max(4, radius // 2 + 3)),
                    0,
                    0,
                    360,
                    (12, 12, 12),
                    -1,
                    cv2.LINE_AA,
                )

            tire_axes = (max(8, radius + 4), max(6, int(radius * 0.72)))
            cv2.ellipse(canvas, (projected_x, projected_y), tire_axes, 0, 0, 360, (18, 18, 20), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (projected_x, projected_y), tire_axes, 0, 0, 360, (92, 92, 96), 2, cv2.LINE_AA)
            cv2.ellipse(canvas, (projected_x, projected_y), (max(5, radius - 2), max(4, int(radius * 0.56))), 0, 0, 360, (50, 52, 56), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (projected_x, projected_y), (max(2, radius // 3), max(2, int(radius * 0.24))), 0, 0, 360, (132, 136, 140), -1, cv2.LINE_AA)

            for spoke_idx in range(6):
                spoke_angle = self._wheel_rotation * 0.7 + spoke_idx * (math.pi / 3.0)
                x1 = projected_x + math.cos(spoke_angle) * max(2, radius * 0.25)
                y1 = projected_y + math.sin(spoke_angle) * max(2, radius * 0.25)
                x2 = projected_x - math.cos(spoke_angle) * max(2, radius * 0.25)
                y2 = projected_y - math.sin(spoke_angle) * max(2, radius * 0.25)
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (220, 220, 220), 1, cv2.LINE_AA)

            cv2.circle(canvas, (projected_x, projected_y), max(2, radius // 5), (255, 210, 90), -1, cv2.LINE_AA)

            axle_line = self._project((self.state.x + offset[0], self.state.y + offset[1], wheel_center[2] + 0.18), camera_pos, camera_target, camera_up, width, height, focal)
            if axle_line is not None:
                cv2.line(canvas, (projected_x, projected_y), (axle_line[0], axle_line[1]), (128, 133, 138), 1, cv2.LINE_AA)

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in edges:
            self._draw_line_3d(canvas, body_world[a], body_world[b], (28, 28, 32), 2, camera_pos, camera_target, camera_up, focal)

        cabin_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in cabin_edges:
            self._draw_line_3d(canvas, cabin_world[a], cabin_world[b], (40, 48, 54), 2, camera_pos, camera_target, camera_up, focal)

        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            self._draw_line_3d(canvas, hood_world[a], hood_world[b], (48, 56, 62), 2, camera_pos, camera_target, camera_up, focal)
            self._draw_line_3d(canvas, roof_world[a], roof_world[b], (150, 170, 180), 1, camera_pos, camera_target, camera_up, focal)

        # Simple shadow and a front marker so the rover reads more clearly.
        shadow_center = (self.state.x + 0.18, self.state.y - 0.12, self.state.z + 0.01)
        shadow = self._project(shadow_center, camera_pos, camera_target, camera_up, width, height, focal)
        if shadow is not None:
            cv2.ellipse(canvas, (shadow[0], shadow[1] + 8), (34, 16), 0, 0, 360, (12, 12, 12), -1, cv2.LINE_AA)

        headlight = self._project(self._world_point(1.05, 0.0, 0.28), camera_pos, camera_target, camera_up, width, height, focal)
        if headlight is not None:
            cv2.circle(canvas, (headlight[0], headlight[1]), 4, (255, 245, 210), -1, cv2.LINE_AA)
            cv2.circle(canvas, (headlight[0], headlight[1]), 8, (255, 220, 120), 1, cv2.LINE_AA)

        sensor_base = (self.state.x - 0.1, self.state.y + 0.02, self.state.z + 0.35)
        sensor_mast_tip = (self.state.x - 0.12, self.state.y + 0.02, self.state.z + 1.18)
        sensor_brace_a = (self.state.x - 0.28, self.state.y - 0.14, self.state.z + 0.42)
        sensor_brace_b = (self.state.x - 0.28, self.state.y + 0.18, self.state.z + 0.42)
        self._draw_line_3d(canvas, sensor_base, sensor_mast_tip, (220, 220, 220), 2, camera_pos, camera_target, camera_up, focal)
        self._draw_line_3d(canvas, sensor_base, sensor_brace_a, (195, 198, 204), 1, camera_pos, camera_target, camera_up, focal)
        self._draw_line_3d(canvas, sensor_base, sensor_brace_b, (195, 198, 204), 1, camera_pos, camera_target, camera_up, focal)

        sensor_body = [
            (sensor_mast_tip[0] - 0.10, sensor_mast_tip[1] - 0.08, sensor_mast_tip[2] - 0.06),
            (sensor_mast_tip[0] + 0.08, sensor_mast_tip[1] - 0.08, sensor_mast_tip[2] - 0.06),
            (sensor_mast_tip[0] + 0.08, sensor_mast_tip[1] + 0.08, sensor_mast_tip[2] - 0.06),
            (sensor_mast_tip[0] - 0.10, sensor_mast_tip[1] + 0.08, sensor_mast_tip[2] - 0.06),
            (sensor_mast_tip[0] - 0.06, sensor_mast_tip[1] - 0.04, sensor_mast_tip[2] + 0.10),
            (sensor_mast_tip[0] + 0.06, sensor_mast_tip[1] - 0.04, sensor_mast_tip[2] + 0.10),
            (sensor_mast_tip[0] + 0.06, sensor_mast_tip[1] + 0.04, sensor_mast_tip[2] + 0.10),
            (sensor_mast_tip[0] - 0.06, sensor_mast_tip[1] + 0.04, sensor_mast_tip[2] + 0.10),
        ]
        self._fill_polygon_3d(canvas, sensor_body, (160, 174, 182), camera_pos, camera_target, camera_up, focal)
        self._draw_line_3d(canvas, sensor_body[0], sensor_body[1], (210, 220, 228), 1, camera_pos, camera_target, camera_up, focal)
        self._draw_line_3d(canvas, sensor_body[1], sensor_body[2], (210, 220, 228), 1, camera_pos, camera_target, camera_up, focal)
        self._draw_line_3d(canvas, sensor_body[2], sensor_body[3], (210, 220, 228), 1, camera_pos, camera_target, camera_up, focal)
        self._draw_line_3d(canvas, sensor_body[3], sensor_body[0], (210, 220, 228), 1, camera_pos, camera_target, camera_up, focal)

        sensor_lens = self._project((sensor_mast_tip[0], sensor_mast_tip[1], sensor_mast_tip[2] + 0.12), camera_pos, camera_target, camera_up, width, height, focal)
        if sensor_lens is not None:
            cv2.circle(canvas, (sensor_lens[0], sensor_lens[1]), 5, (0, 255, 220), -1, cv2.LINE_AA)
            cv2.circle(canvas, (sensor_lens[0], sensor_lens[1]), 8, (120, 255, 220), 1, cv2.LINE_AA)

        self._draw_line_3d(
            canvas,
            (self.state.x, self.state.y, self.state.z + 0.2),
            self._world_point(1.8, 0.0, 0.2),
            (0, 0, 210),
            3,
            camera_pos,
            camera_target,
            camera_up,
            focal,
        )
        self._draw_line_3d(
            canvas,
            (self.state.x, self.state.y, self.state.z + 0.18),
            (self.state.x, self.state.y, self.state.z + 1.3),
            (245, 245, 245),
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
            cv2.circle(canvas, (route[0], route[1]), 7, (0, 255, 180), -1, cv2.LINE_AA)

        self._render_dust(canvas, camera_pos, camera_target, camera_up, focal, width, height)

        haze = np.linspace(0.0, 0.26, height, dtype=np.float32)[:, None]
        haze_mask = np.repeat(haze, width, axis=1)
        haze_rgb = np.array([36, 42, 44], dtype=np.float32)
        canvas = canvas.astype(np.float32)
        canvas = canvas * (1.0 - haze_mask)[:, :, None] + haze_rgb * haze_mask[:, :, None]
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

        panel_w = min(290, max(180, width // 4))
        panel_h = min(290, max(180, height // 3))
        minimap = self._render_minimap((panel_w, panel_h))
        history_panel = self._render_history_panel((panel_w, panel_h))

        x0 = width - panel_w - 14
        minimap_y = 14
        history_y = min(height - panel_h - 14, minimap_y + panel_h + 16)
        canvas[minimap_y:minimap_y + panel_h, x0:x0 + panel_w] = minimap
        canvas[history_y:history_y + panel_h, x0:x0 + panel_w] = history_panel

        hud_lines = [
            f"Target: {target.name if target else 'none'}",
            f"Pose: x={self.state.x:+.2f} m  y={self.state.y:+.2f} m  z={self.state.z:+.2f} m",
            f"Heading: {self.state.heading_deg:+.1f} deg  Speed: {self.state.speed:.2f} m/s",
            f"Mode: {'manual' if self.manual_mode else self.control_mode}  Terrain: {self.terrain_mode}",
            f"pitch {self.state.pitch_deg:+.1f} deg  roll {self.state.roll_deg:+.1f} deg  throttle {self.manual_throttle:+.2f}  steer {self.manual_steer:+.2f}",
        ]
        for idx, line in enumerate(hud_lines):
            y = 28 + idx * 24
            cv2.rectangle(canvas, (14, y - 18), (14 + 560, y + 4), (16, 16, 16), -1)
            cv2.rectangle(canvas, (14, y - 18), (14 + 560, y + 4), (56, 48, 40), 1)
            cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (248, 248, 248), 1, cv2.LINE_AA)

        help_lines = [
            "W/S throttle  A/D steer  M toggle manual  Space pause  R reset  X zero input  1 auto  Q quit",
            "Manual mode drives the rover directly; auto mode follows waypoints.",
        ]
        for idx, line in enumerate(help_lines):
            cv2.putText(canvas, line, (18, height - 72 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (14, height - 98), (width - 320, height - 8), (12, 12, 12), -1)
        cv2.rectangle(canvas, (14, height - 98), (width - 320, height - 8), (62, 56, 48), 1)
        cv2.putText(canvas, "DIRT Rover 3D Simulation", (18, height - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 255, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, "drill  scoop  spectrometer payload  smooth chase cam", (18, height - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)

        if render_scale < 1.0:
            canvas = cv2.resize(canvas, (resolution[0], resolution[1]), interpolation=cv2.INTER_LINEAR)
        return canvas

    def save_path_csv(self, path: str = "sim_rover_3d_path.csv") -> None:
        self.log.save_csv(path)


def run_rover_3d_demo(
    waypoints: Optional[Sequence[Waypoint3D]] = None,
    show_video: bool = True,
    save_csv: bool = True,
    delay_ms: int = 40,
    target_fps: float = 90.0,
    max_steps: int = 420,
    waypoint_tolerance_m: float = 0.45,
    control_mode: str = "pursuit",
    terrain_mode: str = "rolling",
    display_resolution: Tuple[int, int] = (2560, 1440),
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
    if show_video:
        cv2.namedWindow("DIRT Sim - Rover 3D", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DIRT Sim - Rover 3D", display_resolution[0], display_resolution[1])
    log.info("Starting 3D rover demo with %d waypoints", len(sim.waypoints))

    render_scale = 1.0
    if show_video and max(display_resolution) > 1600:
        render_scale = 0.75 if target_fps >= 90.0 else 0.85
    render_width = max(1, int(display_resolution[0] * render_scale))
    render_height = max(1, int(display_resolution[1] * render_scale))
    render_resolution = (render_width, render_height)
    log.info("Display size: %sx%s -> render size: %sx%s (scale %.2fx)", display_resolution[0], display_resolution[1], render_width, render_height, render_scale)

    finished = False
    step = 0
    last_tick = time.perf_counter()
    last_frame = last_tick
    target_fps = float(_clamp(target_fps, 60.0, 120.0))
    frame_interval = 1.0 / target_fps
    display_wait = max(1, min(12, max(1, delay_ms // 4)))
    log.info("Target render FPS: %.1f", target_fps)

    while step < max_steps:
        now = time.perf_counter()
        elapsed = min(0.05, max(0.008, now - last_tick))
        last_tick = now

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

        finished = sim.step(elapsed)
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

        if show_video and (now - last_frame >= frame_interval):
            frame = sim.render(render_resolution)
            if render_resolution != display_resolution:
                frame = cv2.resize(frame, display_resolution, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("DIRT Sim - Rover 3D", frame)
            last_frame = now
            key = cv2.waitKey(display_wait) & 0xFF
            if sim.handle_key(key):
                log.info("Quit by user")
                break

        if finished:
            log.info("3D rover mission complete in %d steps", step + 1)
            break

        step += 1

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
    parser.add_argument("--speed", type=int, default=40, help="Legacy frame delay ms (kept for compatibility)")
    parser.add_argument("--fps", type=float, default=90.0, help="Target render rate in FPS, between 60 and 120")
    parser.add_argument("--resolution", type=str, default="2560x1440", help="Display resolution as WIDTHxHEIGHT, e.g. 2560x1440")
    parser.add_argument("--waypoints", type=str, default=None, help="Optional CSV with columns name,x,y")
    parser.add_argument("--csv", type=str, default="sim_rover_3d_path.csv", help="CSV output path")
    parser.add_argument("--control-mode", choices=sorted(CONTROL_PROFILES.keys()), default="pursuit", help="Motion control profile")
    parser.add_argument("--terrain-mode", choices=sorted(TERRAIN_PROFILES), default="rolling", help="Terrain profile")
    parser.add_argument("--label", type=str, default="demo", help="Run label written into the CSV")
    parser.add_argument("--manual", action="store_true", help="Start in manual driving mode")
    args = parser.parse_args()

    custom_waypoints = _build_waypoints_from_csv(args.waypoints) if args.waypoints else None
    try:
        width, height = (int(v) for v in args.resolution.lower().split("x", 1))
    except ValueError:
        width, height = 2560, 1440
    log.info("Controls: W/S throttle, A/D steer, M manual toggle, Space pause, R reset, X zero, 1 auto, Q quit")
    run_rover_3d_demo(
        waypoints=custom_waypoints,
        show_video=not args.no_gui,
        save_csv=True,
        delay_ms=args.speed,
        target_fps=args.fps,
        max_steps=args.iters,
        control_mode=args.control_mode,
        terrain_mode=args.terrain_mode,
        display_resolution=(width, height),
        csv_path=args.csv,
        run_label=args.label,
        start_manual=args.manual,
    )