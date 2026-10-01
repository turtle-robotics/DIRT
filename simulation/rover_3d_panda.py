"""
DIRT Bot - 3D Rover Simulation (Panda3D backend)
=================================================

This keeps the rover motion model from the existing 3D simulation but moves
the rendering onto Panda3D so the scene can scale better than the OpenCV-only
renderer.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AntialiasAttrib,
    AmbientLight,
    CardMaker,
    ClockObject,
    DirectionalLight,
    Fog,
    Filename,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LVector3f,
    LineSegs,
    Material,
    NodePath,
    PNMImage,
    Texture,
    TextNode,
    TransparencyAttrib,
    WindowProperties,
)


log = logging.getLogger("rover3d-panda")


CONTROL_PROFILES = {
    "smooth": {"max_speed_mps": 0.55, "steering_gain": 1.35, "accel_mps2": 0.28},
    "pursuit": {"max_speed_mps": 0.75, "steering_gain": 2.00, "accel_mps2": 0.45},
    "aggressive": {"max_speed_mps": 1.00, "steering_gain": 2.80, "accel_mps2": 0.62},
}

# NOTE: "farmland" is intentionally included here (not just grassland/rolling/
# dunes/rocky) - _make_terrain() has a dedicated row/furrow/dirt-lane texture
# for it, and Rover3DPandaApp defaults to terrain_mode="farmland". Previously
# "farmland" was missing from this set, so the constructor's own validation
# (`if terrain_mode not in TERRAIN_PROFILES: terrain_mode = "grassland"`)
# silently downgraded every default run to grassland and the farmland texture
# was dead code.
TERRAIN_PROFILES = {"grassland", "rolling", "dunes", "rocky", "farmland"}

TRAIL_MAX_POINTS = 220


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _lerp(value: float, target: float, alpha: float) -> float:
    return value + (target - value) * _clamp(alpha, 0.0, 1.0)


def normalize_resolution(value: Optional[Tuple[int, int] | str]) -> Tuple[int, int]:
    """Normalize the requested window size and keep it in a sane range."""
    if value is None:
        return (2560, 1440)

    if isinstance(value, str):
        try:
            width_text, height_text = value.lower().split("x", 1)
            width = max(640, int(width_text))
            height = max(360, int(height_text))
            return width, height
        except ValueError:
            return (2560, 1440)

    width, height = value
    return max(640, int(width)), max(360, int(height))


def clamp_target_fps(value: float) -> float:
    """Keep the frame target in a plausible smooth-sim range."""
    fps = float(value)
    if fps <= 0:
        return 90.0
    return max(60.0, min(120.0, fps))


def terrain_palette(terrain_mode: str, height: float) -> Tuple[float, float, float]:
    """Return a richer terrain RGB color for a given terrain profile and height."""
    h = float(height)
    if terrain_mode == "dunes":
        if h < -0.05:
            return (0.42, 0.36, 0.24)
        if h < 0.30:
            return (0.71, 0.57, 0.36)
        return (0.94, 0.80, 0.52)
    if terrain_mode == "rocky":
        if h < 0.10:
            return (0.24, 0.22, 0.20)
        if h < 0.45:
            return (0.43, 0.40, 0.35)
        return (0.62, 0.58, 0.51)
    if terrain_mode == "farmland":
        if h < -0.08:
            return (0.28, 0.20, 0.12)
        if h < 0.12:
            return (0.52, 0.39, 0.20)
        if h < 0.42:
            return (0.62, 0.51, 0.23)
        return (0.72, 0.66, 0.32)
    if terrain_mode == "grassland":
        if h < 0.0:
            return (0.20, 0.26, 0.18)
        if h < 0.30:
            return (0.36, 0.46, 0.25)
        return (0.57, 0.65, 0.38)
    if h < 0.0:
        return (0.26, 0.24, 0.22)
    if h < 0.35:
        return (0.44, 0.40, 0.30)
    return (0.78, 0.70, 0.55)


def lighting_profile(terrain_mode: str) -> dict:
    """Return stronger contrast lighting defaults tuned to the scene."""
    if terrain_mode == "farmland":
        return {
            "ambient": (0.26, 0.27, 0.30, 1.0),
            "sun_color": (1.00, 0.88, 0.72, 1.0),
            "fill_color": (0.16, 0.24, 0.42, 1.0),
            "background": (0.54, 0.68, 0.74, 1.0),
            "fog_color": (0.56, 0.72, 0.80),
            "fog_density": 0.012,
        }
    if terrain_mode == "dunes":
        return {
            "ambient": (0.42, 0.34, 0.30, 1.0),
            "sun_color": (1.00, 0.92, 0.78, 1.0),
            "fill_color": (0.22, 0.20, 0.32, 1.0),
            "background": (0.72, 0.76, 0.70, 1.0),
            "fog_color": (0.72, 0.78, 0.74),
            "fog_density": 0.009,
        }
    if terrain_mode == "rocky":
        return {
            "ambient": (0.30, 0.28, 0.32, 1.0),
            "sun_color": (0.96, 0.90, 0.82, 1.0),
            "fill_color": (0.18, 0.20, 0.28, 1.0),
            "background": (0.48, 0.52, 0.58, 1.0),
            "fog_color": (0.60, 0.64, 0.70),
            "fog_density": 0.011,
        }
    return {
        "ambient": (0.32, 0.34, 0.38, 1.0),
        "sun_color": (1.00, 0.93, 0.80, 1.0),
        "fill_color": (0.22, 0.30, 0.42, 1.0),
        "background": (0.54, 0.68, 0.74, 1.0),
        "fog_color": (0.56, 0.70, 0.74),
        "fog_density": 0.010,
    }


def weather_profile(terrain_mode: str) -> dict:
    """Return a richer atmospheric profile with haze, glow, and cloud depth."""
    if terrain_mode == "farmland":
        return {
            "mist_color": (0.78, 0.82, 0.85),
            "mist_density": 0.016,
            "sun_glow": (1.00, 0.72, 0.34, 0.88),
            "cloud_alpha": 0.28,
            "sun_bloom": 1.12,
        }
    if terrain_mode == "dunes":
        return {
            "mist_color": (0.86, 0.82, 0.76),
            "mist_density": 0.013,
            "sun_glow": (1.00, 0.86, 0.60, 0.82),
            "cloud_alpha": 0.22,
            "sun_bloom": 1.08,
        }
    if terrain_mode == "rocky":
        return {
            "mist_color": (0.72, 0.76, 0.80),
            "mist_density": 0.015,
            "sun_glow": (0.96, 0.82, 0.58, 0.80),
            "cloud_alpha": 0.18,
            "sun_bloom": 1.06,
        }
    return {
        "mist_color": (0.72, 0.78, 0.82),
        "mist_density": 0.014,
        "sun_glow": (1.00, 0.90, 0.72, 0.86),
        "cloud_alpha": 0.24,
        "sun_bloom": 1.10,
    }


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
    def __init__(self, span_m: float = 48.0, terrain_mode: str = "grassland"):
        self.span_m = span_m
        self.terrain_mode = terrain_mode if terrain_mode in TERRAIN_PROFILES else "grassland"

    def height(self, x, y):
        """Height at (x, y). Accepts scalars or numpy arrays (broadcast)."""
        if self.terrain_mode == "grassland":
            base = 0.015 * np.sin(x / 6.8) + 0.012 * np.cos(y / 7.6)
            brush = 0.008 * np.sin((x + y) / 5.8)
            track = 0.006 * np.cos((x - 0.6 * y) / 6.5)
            return base + brush + track
        if self.terrain_mode == "dunes":
            return 0.55 * np.sin(x / 2.4) + 0.25 * np.sin((x + y) / 3.6) + 0.14 * np.cos(y / 4.4)
        if self.terrain_mode == "rocky":
            return (
                0.35 * np.sin(x / 1.8)
                + 0.25 * np.cos(y / 2.2)
                + 0.12 * np.sin((x * 1.7 + y * 1.3) / 1.6)
                + 0.08 * np.cos(np.hypot(x, y) / 1.4)
            )
        return (
            0.45 * np.sin(x / 2.9)
            + 0.35 * np.cos(y / 3.4)
            + 0.18 * np.sin((x + y) / 4.5)
            + 0.08 * np.cos(np.hypot(x, y) / 2.7)
        )

    def gradient(self, x: float, y: float, delta: float = 0.15) -> Tuple[float, float]:
        dx = (self.height(x + delta, y) - self.height(x - delta, y)) / (2.0 * delta)
        dy = (self.height(x, y + delta) - self.height(x, y - delta)) / (2.0 * delta)
        return dx, dy

    def height_and_gradient_grid(
        self, xs: np.ndarray, ys: np.ndarray, delta: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized height + gradient over a full (len(ys), len(xs)) grid.

        Used by _make_terrain to build the whole mesh with a handful of
        numpy calls instead of one Python-level height()/gradient() call
        per vertex (which, at resolution=48, was ~7000 scalar trig
        evaluations run through the interpreter on every scene build).
        """
        gx, gy = np.meshgrid(xs, ys)
        z = self.height(gx, gy)
        slope_x = (self.height(gx + delta, gy) - self.height(gx - delta, gy)) / (2.0 * delta)
        slope_y = (self.height(gx, gy + delta) - self.height(gx, gy - delta)) / (2.0 * delta)
        return z, slope_x, slope_y


class Rover3DPandaApp(ShowBase):
    def __init__(
        self,
        waypoints: Optional[Sequence[Waypoint3D]] = None,
        terrain_span_m: float = 48.0,
        waypoint_tolerance_m: float = 0.45,
        max_speed_mps: Optional[float] = None,
        steering_gain: Optional[float] = None,
        accel_mps2: Optional[float] = None,
        control_mode: str = "pursuit",
        terrain_mode: str = "farmland",
        run_label: str = "demo",
        max_steps: Optional[int] = None,
        display_resolution: Optional[Tuple[int, int] | str] = (2560, 1440),
        target_fps: float = 90.0,
    ):
        super().__init__()

        if control_mode not in CONTROL_PROFILES:
            control_mode = "pursuit"
        if terrain_mode not in TERRAIN_PROFILES:
            terrain_mode = "grassland"

        profile = CONTROL_PROFILES[control_mode]
        self.control_mode = control_mode
        self.max_speed_mps = profile["max_speed_mps"] if max_speed_mps is None else max_speed_mps
        self.steering_gain = profile["steering_gain"] if steering_gain is None else steering_gain
        self.accel_mps2 = profile["accel_mps2"] if accel_mps2 is None else accel_mps2

        self.terrain_mode = terrain_mode
        self.terrain = TerrainField(span_m=terrain_span_m, terrain_mode=terrain_mode)
        self.waypoint_tolerance_m = waypoint_tolerance_m
        self.waypoints = list(waypoints or self._default_waypoints())
        self.run_label = run_label
        self.max_steps = max_steps
        self.display_resolution = normalize_resolution(display_resolution)
        self.target_fps = clamp_target_fps(target_fps)
        self.mission_mode = True
        self.mission_phase = "drive_to_drill"
        self.mission_hold_timer = 0.0
        self._drill_lift = 0.0
        self._measure_lift = 0.0
        self._drill_spin = 0.0
        self._mission_measure_target: Optional[Tuple[float, float]] = None
        self._drill_cycle_complete = False
        self._measurement_complete = False

        self.state = RoverState(x=-8.0, y=-7.0, heading_deg=25.0)
        self.state.z = self.terrain.height(self.state.x, self.state.y)
        self.target_idx = 0
        self.step_idx = 0
        self.finished = False
        self._quit_requested = False
        self._camera_pos = LVector3f(-25.0, -27.0, self.state.z + 18.0)
        self._camera_target = LVector3f(0.0, 0.0, 0.0)
        self._render_frame = 0
        self._wheel_rotation = 0.0
        self._trail_points: Deque[Tuple[float, float, float]] = deque(maxlen=TRAIL_MAX_POINTS)
        self._trail_dirty = True
        self.log = MotionLog()

        self.manual_mode = False
        self.paused = False
        self.manual_throttle = 0.0
        self.manual_steer = 0.0
        self._keys = {"w": False, "a": False, "s": False, "d": False}
        self._trail_np: Optional[NodePath] = None
        self._sky_np: Optional[NodePath] = None
        self._sun_np: Optional[NodePath] = None
        self._cloud_nodes: List[NodePath] = []
        self._obstacle_nodes: List[Tuple[str, NodePath, float, float, float]] = []
        self._detected_obstacles: set[str] = set()
        self._nearest_obstacle_text: str = "none"
        self._shadow_np: Optional[NodePath] = None
        self._drill_stage_np: Optional[NodePath] = None
        self._measure_stage_np: Optional[NodePath] = None
        self._field_rows: List[NodePath] = []
        self._grass_nodes: List[NodePath] = []
        self._tread_nodes: List[Tuple[NodePath, float, float]] = []
        self._route_np: Optional[NodePath] = None
        self._camera_mode = "follow"
        self._snapshot_path: Optional[str] = None
        self._snapshot_taken = False
        self._snapshot_exit = False

        self.disableMouse()
        self.setBackgroundColor(0.54, 0.68, 0.74, 1.0)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.camLens.setFov(62)
        self.camLens.setNearFar(0.15, 2000.0)
        self.setFrameRateMeter(False)
        self._apply_display_settings()
        self._build_scene()
        self._build_hud()
        self._bind_controls()
        self._sync_scene()

        clock = ClockObject.getGlobalClock()
        clock.setMode(ClockObject.MLimited)
        try:
            clock.setFrameRate(self.target_fps)
        except Exception:
            pass

        self.taskMgr.add(self._update_task, "rover3d-update")

    def _apply_display_settings(self) -> None:
        if self.win is None:
            return

        props = WindowProperties()
        props.setSize(self.display_resolution[0], self.display_resolution[1])
        props.setFullscreen(False)
        props.setTitle(f"DIRT Panda Rover - {self.display_resolution[0]}x{self.display_resolution[1]} @ {self.target_fps:.0f} FPS")
        try:
            props.setMinimumSize(1280, 720)
        except Exception:
            pass
        self.win.requestProperties(props)

    def _default_waypoints(self) -> List[Waypoint3D]:
        return [
            Waypoint3D("Start", -8.0, -7.0),
            Waypoint3D("Sample A", -1.5, -2.5),
            Waypoint3D("Sample B", 4.0, 1.5),
            Waypoint3D("Sample C", 8.0, 6.0),
        ]

    def _bind_controls(self) -> None:
        self.accept("escape", self._request_quit)
        self.accept("q", self._request_quit)
        self.accept("m", self.toggle_manual_mode)
        self.accept("space", self.toggle_pause)
        self.accept("r", self.reset_pose)
        self.accept("x", self.zero_manual_input)
        self.accept("1", self._set_auto_mode)
        self.accept("2", self._toggle_mission_mode)
        self.accept("c", self._cycle_camera)
        self.accept("f12", self.capture_snapshot)
        for key in ["w", "a", "s", "d"]:
            self.accept(key, self._set_key, [key, True])
            self.accept(f"{key}-up", self._set_key, [key, False])

    def _set_key(self, key: str, value: bool) -> None:
        self._keys[key] = value

    def _set_auto_mode(self) -> None:
        self.manual_mode = False
        self.manual_throttle = 0.0
        self.manual_steer = 0.0

    def _request_quit(self) -> None:
        self._quit_requested = True

    def _toggle_mission_mode(self) -> None:
        self.mission_mode = not self.mission_mode
        self.mission_phase = "drive_to_drill"
        self.mission_hold_timer = 0.0
        self._drill_lift = 0.0
        self._measure_lift = 0.0
        self._mission_measure_target = None
        self._drill_cycle_complete = False
        self._measurement_complete = False

    def _cycle_camera(self) -> None:
        modes = ("follow", "overhead", "cockpit")
        self._camera_mode = modes[(modes.index(self._camera_mode) + 1) % len(modes)]

    def _build_box(self, name: str, color: Tuple[float, float, float, float]) -> NodePath:
        format_ = GeomVertexFormat.getV3n3()
        format_ = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData(name, format_, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        tris = GeomTriangles(Geom.UHStatic)

        faces = [
            ((0.0, 0.0, 1.0), [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]),
            ((0.0, 0.0, -1.0), [(-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5), (-0.5, -0.5, -0.5)]),
            ((0.0, 1.0, 0.0), [(-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5)]),
            ((0.0, -1.0, 0.0), [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)]),
            ((1.0, 0.0, 0.0), [(0.5, -0.5, 0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)]),
            ((-1.0, 0.0, 0.0), [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5)]),
        ]

        for face_index, (face_normal, corners) in enumerate(faces):
            base = face_index * 4
            for corner in corners:
                vertex.addData3f(*corner)
                normal.addData3f(*face_normal)
            tris.addVertices(base, base + 1, base + 2)
            tris.addVertices(base, base + 2, base + 3)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode(name)
        node.addGeom(geom)
        np_node = NodePath(node)
        np_node.setColor(*color)
        return np_node

    def _make_terrain(self, resolution: int = 48) -> NodePath:
        """Build the terrain mesh.

        Height/normal/color for all resolution*resolution vertices are
        computed with a handful of vectorized numpy calls (see
        TerrainField.height_and_gradient_grid) instead of one Python
        function call per vertex per axis. GeomVertexWriter still needs a
        per-vertex write (Panda3D's API has no bulk-array insert), but the
        expensive trig math that used to run at the interpreter level now
        runs once, vectorized, before that loop.
        """
        format_ = GeomVertexFormat.getV3n3c4t2()
        vdata = GeomVertexData("terrain", format_, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        color = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHStatic)

        half = self.terrain.span_m / 2.0
        xs = np.linspace(-half, half, resolution)
        ys = np.linspace(-half, half, resolution)

        z_grid, slope_x_grid, slope_y_grid = self.terrain.height_and_gradient_grid(xs, ys)

        nx_grid = -slope_x_grid
        ny_grid = -slope_y_grid
        nz_grid = np.ones_like(z_grid)
        length_grid = np.sqrt(nx_grid**2 + ny_grid**2 + nz_grid**2)
        length_grid[length_grid == 0] = 1.0
        nx_grid /= length_grid
        ny_grid /= length_grid
        nz_grid /= length_grid

        height_t_grid = np.clip((z_grid + 0.8) / 1.7, 0.0, 1.0)

        gx, gy = np.meshgrid(xs, ys)

        if self.terrain_mode == "farmland":
            row_band = 0.5 + 0.5 * np.sin(gy * 5.4)
            furrow_band = 0.5 + 0.5 * np.sin((gx * 3.0 + gy * 0.24) * 2.8)
            dirt_lane = 0.5 + 0.5 * np.sin((gx - 0.28 * gy) * 1.7)
            band_mix = 0.56 * row_band + 0.28 * furrow_band + 0.16 * dirt_lane

            low_mask = band_mix < 0.44
            mid_mask = (band_mix >= 0.44) & (band_mix < 0.72)
            high_mask = band_mix >= 0.72

            base_grid = np.empty(z_grid.shape + (3,), dtype=float)
            top_grid = np.empty(z_grid.shape + (3,), dtype=float)
            base_grid[low_mask] = (0.25, 0.19, 0.11)
            top_grid[low_mask] = (0.41, 0.32, 0.15)
            base_grid[mid_mask] = (0.38, 0.28, 0.15)
            top_grid[mid_mask] = (0.60, 0.50, 0.24)
            base_grid[high_mask] = (0.30, 0.22, 0.12)
            top_grid[high_mask] = (0.74, 0.67, 0.32)
        elif self.terrain_mode == "dunes":
            base_grid = np.tile((0.60, 0.48, 0.28), z_grid.shape + (1,))
            top_grid = np.tile((0.98, 0.84, 0.56), z_grid.shape + (1,))
        elif self.terrain_mode == "rocky":
            base_grid = np.tile((0.26, 0.25, 0.24), z_grid.shape + (1,))
            top_grid = np.tile((0.66, 0.62, 0.56), z_grid.shape + (1,))
        else:
            base_grid = np.tile((0.22, 0.28, 0.18), z_grid.shape + (1,))
            top_grid = np.tile((0.74, 0.76, 0.42), z_grid.shape + (1,))

        height_t_3 = height_t_grid[..., None]
        rgb_grid = base_grid * (1.0 - height_t_3) + top_grid * height_t_3

        # Add a subtle warm highlight to the higher terrain values to preserve a
        # richer field look without over-saturating the whole mesh.
        if self.terrain_mode in {"farmland", "grassland"}:
            rgb_grid = rgb_grid * 0.94 + np.stack(
                [
                    np.clip(height_t_grid * 0.18, 0.0, 1.0),
                    np.clip(height_t_grid * 0.12, 0.0, 1.0),
                    np.clip(height_t_grid * 0.06, 0.0, 1.0),
                ],
                axis=-1,
            )

        index_grid: List[List[int]] = []
        index = 0
        for row in range(resolution):
            row_indices: List[int] = []
            for col in range(resolution):
                vertex.addData3f(float(gx[row, col]), float(gy[row, col]), float(z_grid[row, col]))
                normal.addData3f(float(nx_grid[row, col]), float(ny_grid[row, col]), float(nz_grid[row, col]))
                texcoord.addData2f(float(gx[row, col] / 3.2), float(gy[row, col] / 3.2))
                r, g, b = rgb_grid[row, col]
                color.addData4f(float(r), float(g), float(b), 1.0)
                row_indices.append(index)
                index += 1
            index_grid.append(row_indices)

        for row in range(resolution - 1):
            for col in range(resolution - 1):
                a = index_grid[row][col]
                b = index_grid[row][col + 1]
                c = index_grid[row + 1][col + 1]
                d = index_grid[row + 1][col]
                tris.addVertices(a, b, c)
                tris.addVertices(a, c, d)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("terrain")
        node.addGeom(geom)
        terrain_np = self.render.attachNewNode(node)
        terrain_np.setTwoSided(True)
        terrain_np.setTexture(self._make_grass_texture(), 1)
        return terrain_np

    def _build_waypoint_marker(self, colour: Tuple[float, float, float, float]) -> NodePath:
        marker = self._build_box("waypoint", colour)
        marker.setScale(0.18, 0.18, 0.55)
        return marker

    def _apply_material(
        self,
        node: NodePath,
        diffuse: Tuple[float, float, float, float],
        ambient: Tuple[float, float, float, float],
        specular: Tuple[float, float, float, float],
        shininess: float,
    ) -> None:
        material = Material()
        material.setDiffuse(diffuse)
        material.setAmbient(ambient)
        material.setSpecular(specular)
        material.setShininess(shininess)
        node.setMaterial(material, 1)

    def _make_rover_shadow(self) -> NodePath:
        card = CardMaker("rover-shadow")
        card.setFrame(-1.6, 1.6, -0.9, 0.9)
        shadow = self.render.attachNewNode(card.generate())
        shadow.setP(-90)
        shadow.setTransparency(TransparencyAttrib.MAlpha)
        shadow.setColor(0.0, 0.0, 0.0, 0.34)
        shadow.setDepthWrite(False)
        shadow.setBin("transparent", 10)
        shadow.setScale(1.0, 1.0, 1.0)
        return shadow

    def _make_soft_disc_texture(
        self, name: str, core: Tuple[float, float, float], edge: Tuple[float, float, float, float]
    ) -> Texture:
        """Radial gradient disc used for the sun/cloud sprites.

        The alpha falloff and RGBA blend for all 128*128 pixels are computed
        with numpy in a few vector ops. PNMImage has no bulk-array setter,
        so writing the result still needs one setXelA() call per pixel, but
        that loop is now pure array indexing instead of also doing the
        sqrt/smoothstep/lerp math per pixel in the interpreter.
        """
        size = 128
        idx = np.arange(size)
        gx, gy = np.meshgrid(idx, idx)
        dx = (gx - 63.5) / 63.5
        dy = (gy - 63.5) / 63.5
        dist = np.sqrt(dx**2 + dy**2)
        alpha = np.clip(1.0 - dist, 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        r = edge[0] * (1.0 - alpha) + core[0] * alpha
        g = edge[1] * (1.0 - alpha) + core[1] * alpha
        b = edge[2] * (1.0 - alpha) + core[2] * alpha
        a = edge[3] * (1.0 - alpha) + alpha

        image = PNMImage(size, size, 4)
        image.fill(0.0, 0.0, 0.0)
        image.alphaFill(0.0)
        for y in range(size):
            for x in range(size):
                image.setXelA(x, y, float(r[y, x]), float(g[y, x]), float(b[y, x]), float(a[y, x]))

        texture = Texture(name)
        texture.load(image)
        texture.setMagfilter(Texture.FTLinear)
        texture.setMinfilter(Texture.FTLinearMipmapLinear)
        return texture

    def _make_grass_texture(self) -> Texture:
        """Create a small tiled grass and soil map without an external asset."""
        size = 256
        rng = np.random.default_rng(17)
        noise = rng.random((size, size))
        coarse = rng.random((32, 32))
        coarse = np.repeat(np.repeat(coarse, 8, axis=0), 8, axis=1)
        green = np.clip(0.58 + noise * 0.18 + coarse * 0.16, 0.0, 1.0)
        soil = noise > 0.965
        blade = noise < 0.035

        image = PNMImage(size, size, 3)
        for y in range(size):
            for x in range(size):
                value = green[y, x]
                if soil[y, x]:
                    rgb = (0.24 + value * 0.10, 0.19 + value * 0.08, 0.08 + value * 0.04)
                elif blade[y, x]:
                    rgb = (0.18 + value * 0.10, 0.34 + value * 0.18, 0.08 + value * 0.05)
                else:
                    rgb = (0.25 + value * 0.14, 0.42 + value * 0.20, 0.10 + value * 0.06)
                image.setXel(x, y, *rgb)

        texture = Texture("grass-map")
        texture.load(image)
        texture.setWrapU(Texture.WMRepeat)
        texture.setWrapV(Texture.WMRepeat)
        texture.setMinfilter(Texture.FTLinearMipmapLinear)
        texture.setMagfilter(Texture.FTLinear)
        return texture

    def _make_billboard_sprite(
        self,
        name: str,
        texture: Texture,
        scale: float,
        pos: Tuple[float, float, float],
        color: Tuple[float, float, float, float],
    ) -> NodePath:
        card = CardMaker(name)
        card.setFrame(-1.0, 1.0, -1.0, 1.0)
        node = self.render.attachNewNode(card.generate())
        node.setBillboardPointEye()
        node.setPos(*pos)
        node.setScale(scale)
        node.setTexture(texture)
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setColor(*color)
        node.setDepthWrite(False)
        node.setBin("transparent", 20)
        return node

    def _make_sky_dome(self, radius: float = 140.0, slices: int = 48, stacks: int = 24) -> NodePath:
        format_ = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData("sky", format_, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        color = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHStatic)

        for stack in range(stacks + 1):
            phi = (stack / stacks) * (math.pi / 2.0)
            z = math.cos(phi)
            ring = math.sin(phi)
            mix = stack / max(stacks, 1)
            for slice_idx in range(slices + 1):
                theta = (slice_idx / slices) * (math.pi * 2.0)
                x = math.cos(theta) * ring
                y = math.sin(theta) * ring
                nx, ny, nz = -x, -y, -z
                vertex.addData3f(x * radius, y * radius, z * radius)
                normal.addData3f(nx, ny, nz)
                if mix < 0.42:
                    t = mix / 0.42
                    r = 0.92 * (1.0 - t) + 0.55 * t
                    g = 0.55 * (1.0 - t) + 0.74 * t
                    b = 0.28 * (1.0 - t) + 0.95 * t
                else:
                    t = (mix - 0.42) / 0.58
                    r = 0.55 * (1.0 - t) + 0.16 * t
                    g = 0.74 * (1.0 - t) + 0.28 * t
                    b = 0.95 * (1.0 - t) + 0.22 * t
                color.addData4f(r, g, b, 1.0)

        row_width = slices + 1
        for stack in range(stacks):
            for slice_idx in range(slices):
                a = stack * row_width + slice_idx
                b = a + 1
                c = a + row_width + 1
                d = a + row_width
                tris.addVertices(a, b, c)
                tris.addVertices(a, c, d)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("sky")
        node.addGeom(geom)
        sky_np = self.render.attachNewNode(node)
        sky_np.setTwoSided(True)
        sky_np.setLightOff(1)
        sky_np.setDepthWrite(False)
        sky_np.setBin("background", 0)
        return sky_np

    def _build_rover(self) -> NodePath:
        root = self.render.attachNewNode("rover")

        deck = self._build_box("solar-deck", (0.08, 0.10, 0.12, 1.0))
        deck.reparentTo(root)
        deck.setScale(1.88, 1.02, 0.04)
        deck.setPos(-0.02, 0.0, 0.92)
        self._apply_material(deck, (0.06, 0.08, 0.10, 1.0), (0.03, 0.04, 0.05, 1.0), (0.10, 0.12, 0.14, 1.0), 6.0)

        for idx, x_pos in enumerate([-0.72, -0.24, 0.24, 0.72]):
            tile = self._build_box(f"solar-tile-{idx}", (0.04, 0.05, 0.07, 1.0))
            tile.reparentTo(root)
            tile.setScale(0.22, 0.86, 0.010)
            tile.setPos(x_pos, 0.0, 0.965)
            self._apply_material(tile, (0.05, 0.06, 0.08, 1.0), (0.02, 0.03, 0.04, 1.0), (0.08, 0.10, 0.12, 1.0), 2.0)

        top_spine = self._build_box("top-spine", (0.72, 0.74, 0.78, 1.0))
        top_spine.reparentTo(root)
        top_spine.setScale(1.32, 0.22, 0.06)
        top_spine.setPos(-0.02, 0.0, 0.78)
        self._apply_material(top_spine, (0.70, 0.72, 0.76, 1.0), (0.22, 0.24, 0.26, 1.0), (0.88, 0.90, 0.92, 1.0), 26.0)

        frame_front = self._build_box("deck-frame-front", (0.24, 0.20, 0.16, 1.0))
        frame_front.reparentTo(root)
        frame_front.setScale(1.94, 0.06, 0.04)
        frame_front.setPos(-0.02, 0.52, 0.92)
        self._apply_material(frame_front, (0.18, 0.16, 0.12, 1.0), (0.08, 0.07, 0.05, 1.0), (0.24, 0.20, 0.16, 1.0), 10.0)

        frame_back = self._build_box("deck-frame-back", (0.24, 0.20, 0.16, 1.0))
        frame_back.reparentTo(root)
        frame_back.setScale(1.94, 0.06, 0.04)
        frame_back.setPos(-0.02, -0.52, 0.92)
        self._apply_material(frame_back, (0.18, 0.16, 0.12, 1.0), (0.08, 0.07, 0.05, 1.0), (0.24, 0.20, 0.16, 1.0), 10.0)

        nose = self._build_box("front-nose", (0.16, 0.18, 0.22, 1.0))
        nose.reparentTo(root)
        nose.setScale(0.44, 0.28, 0.18)
        nose.setPos(1.02, 0.0, 0.56)
        self._apply_material(nose, (0.18, 0.20, 0.24, 1.0), (0.08, 0.09, 0.10, 1.0), (0.36, 0.40, 0.44, 1.0), 30.0)

        front_beam = self._build_box("front-beam", (0.22, 0.24, 0.28, 1.0))
        front_beam.reparentTo(root)
        front_beam.setScale(0.48, 0.16, 0.10)
        front_beam.setPos(0.86, 0.0, 0.40)
        self._apply_material(front_beam, (0.22, 0.24, 0.28, 1.0), (0.08, 0.09, 0.10, 1.0), (0.40, 0.44, 0.48, 1.0), 24.0)

        center_body = self._build_box("center-body", (0.20, 0.22, 0.25, 1.0))
        center_body.reparentTo(root)
        center_body.setScale(1.30, 0.76, 0.20)
        center_body.setPos(0.00, 0.0, 0.42)
        self._apply_material(center_body, (0.18, 0.20, 0.22, 1.0), (0.08, 0.09, 0.10, 1.0), (0.36, 0.40, 0.44, 1.0), 28.0)

        belly_pan = self._build_box("belly-pan", (0.08, 0.09, 0.11, 1.0))
        belly_pan.reparentTo(root)
        belly_pan.setScale(1.08, 0.60, 0.06)
        belly_pan.setPos(0.00, 0.0, 0.22)
        self._apply_material(belly_pan, (0.08, 0.09, 0.11, 1.0), (0.04, 0.05, 0.05, 1.0), (0.16, 0.18, 0.20, 1.0), 12.0)

        left_sponson = self._build_box("left-sponson", (0.18, 0.20, 0.22, 1.0))
        left_sponson.reparentTo(root)
        left_sponson.setScale(0.60, 0.14, 0.10)
        left_sponson.setPos(-0.02, 0.54, 0.28)
        self._apply_material(left_sponson, (0.18, 0.20, 0.22, 1.0), (0.08, 0.09, 0.10, 1.0), (0.34, 0.38, 0.42, 1.0), 18.0)

        right_sponson = self._build_box("right-sponson", (0.18, 0.20, 0.22, 1.0))
        right_sponson.reparentTo(root)
        right_sponson.setScale(0.60, 0.14, 0.10)
        right_sponson.setPos(-0.02, -0.54, 0.28)
        self._apply_material(right_sponson, (0.18, 0.20, 0.22, 1.0), (0.08, 0.09, 0.10, 1.0), (0.34, 0.38, 0.42, 1.0), 18.0)

        rear_box = self._build_box("rear-box", (0.16, 0.20, 0.24, 1.0))
        rear_box.reparentTo(root)
        rear_box.setScale(0.66, 0.52, 0.22)
        rear_box.setPos(-0.92, 0.0, 0.60)
        self._apply_material(rear_box, (0.18, 0.22, 0.26, 1.0), (0.08, 0.10, 0.12, 1.0), (0.44, 0.48, 0.52, 1.0), 40.0)

        spectrometer_box = self._build_box("spectrometer-box", (0.22, 0.26, 0.30, 1.0))
        spectrometer_box.reparentTo(root)
        spectrometer_box.setScale(0.30, 0.20, 0.14)
        spectrometer_box.setPos(-0.88, 0.0, 0.72)
        self._apply_material(spectrometer_box, (0.22, 0.26, 0.30, 1.0), (0.08, 0.10, 0.12, 1.0), (0.50, 0.56, 0.60, 1.0), 44.0)

        spectrometer_lid = self._build_box("spectrometer-lid", (0.30, 0.36, 0.42, 1.0))
        spectrometer_lid.reparentTo(root)
        spectrometer_lid.setScale(0.30, 0.20, 0.05)
        spectrometer_lid.setPos(-0.88, 0.0, 0.88)
        self._apply_material(spectrometer_lid, (0.32, 0.38, 0.44, 1.0), (0.10, 0.12, 0.14, 1.0), (0.56, 0.60, 0.66, 1.0), 56.0)

        stop_bracket = self._build_box("stop-bracket", (0.24, 0.24, 0.26, 1.0))
        stop_bracket.reparentTo(root)
        stop_bracket.setScale(0.09, 0.24, 0.16)
        stop_bracket.setPos(-1.18, 0.0, 0.46)
        self._apply_material(stop_bracket, (0.22, 0.22, 0.24, 1.0), (0.08, 0.08, 0.08, 1.0), (0.34, 0.36, 0.38, 1.0), 20.0)

        mast_base = self._build_box("mast-base", (0.80, 0.80, 0.84, 1.0))
        mast_base.reparentTo(root)
        mast_base.setScale(0.07, 0.07, 0.20)
        mast_base.setPos(-0.06, 0.0, 1.00)
        self._apply_material(mast_base, (0.82, 0.82, 0.86, 1.0), (0.30, 0.30, 0.32, 1.0), (0.62, 0.64, 0.66, 1.0), 32.0)

        mast = self._build_box("mast", (0.88, 0.88, 0.90, 1.0))
        mast.reparentTo(root)
        mast.setScale(0.05, 0.05, 0.72)
        mast.setPos(-0.06, 0.0, 1.34)
        self._apply_material(mast, (0.92, 0.92, 0.94, 1.0), (0.34, 0.34, 0.36, 1.0), (0.68, 0.70, 0.72, 1.0), 42.0)

        mast_head = self._build_box("mast_head", (0.12, 0.94, 0.80, 1.0))
        mast_head.reparentTo(root)
        mast_head.setScale(0.12, 0.12, 0.12)
        mast_head.setPos(-0.06, 0.0, 1.84)
        self._apply_material(mast_head, (0.18, 0.95, 0.82, 1.0), (0.04, 0.18, 0.16, 1.0), (0.60, 1.0, 0.92, 1.0), 70.0)

        mast_sensor = self._build_box("mast-sensor", (0.20, 0.96, 0.84, 1.0))
        mast_sensor.reparentTo(root)
        mast_sensor.setScale(0.12, 0.12, 0.05)
        mast_sensor.setPos(-0.06, 0.0, 1.96)
        self._apply_material(mast_sensor, (0.12, 0.92, 0.82, 1.0), (0.04, 0.18, 0.16, 1.0), (0.70, 1.0, 0.94, 1.0), 90.0)

        camera_pod = self._build_box("camera-pod", (0.12, 0.15, 0.18, 1.0))
        camera_pod.reparentTo(root)
        camera_pod.setScale(0.18, 0.14, 0.10)
        camera_pod.setPos(0.28, 0.0, 0.98)
        self._apply_material(camera_pod, (0.14, 0.16, 0.18, 1.0), (0.06, 0.07, 0.08, 1.0), (0.46, 0.50, 0.55, 1.0), 48.0)

        camera_lens = self._build_box("camera-lens", (0.08, 0.92, 0.88, 1.0))
        camera_lens.reparentTo(root)
        camera_lens.setScale(0.06, 0.06, 0.06)
        camera_lens.setPos(0.42, 0.0, 0.98)
        self._apply_material(camera_lens, (0.10, 0.90, 0.86, 1.0), (0.05, 0.22, 0.20, 1.0), (0.85, 1.0, 1.0, 1.0), 96.0)

        front_light_left = self._build_box("front-light-left", (1.0, 0.86, 0.52, 1.0))
        front_light_left.reparentTo(root)
        front_light_left.setScale(0.08, 0.05, 0.05)
        front_light_left.setPos(0.94, 0.22, 0.18)
        self._apply_material(front_light_left, (1.0, 0.90, 0.58, 1.0), (0.16, 0.12, 0.08, 1.0), (1.0, 0.98, 0.84, 1.0), 120.0)

        front_light_right = self._build_box("front-light-right", (1.0, 0.86, 0.52, 1.0))
        front_light_right.reparentTo(root)
        front_light_right.setScale(0.08, 0.05, 0.05)
        front_light_right.setPos(0.94, -0.22, 0.18)
        self._apply_material(front_light_right, (1.0, 0.90, 0.58, 1.0), (0.16, 0.12, 0.08, 1.0), (1.0, 0.98, 0.84, 1.0), 120.0)

        for side_sign, side_name, y_pos in [(1.0, "left", 0.94), (-1.0, "right", -0.94)]:
            side_mount = self._build_box(f"side-mount-{side_name}", (0.22, 0.24, 0.28, 1.0))
            side_mount.reparentTo(root)
            side_mount.setScale(0.12, 0.20, 0.12)
            side_mount.setPos(0.64, y_pos * 0.66, 0.28)
            self._apply_material(side_mount, (0.22, 0.24, 0.28, 1.0), (0.08, 0.09, 0.10, 1.0), (0.38, 0.42, 0.46, 1.0), 24.0)

            side_strut = self._build_box(f"side-strut-{side_name}", (0.18, 0.20, 0.24, 1.0))
            side_strut.reparentTo(root)
            side_strut.setScale(0.62, 0.08, 0.08)
            side_strut.setPos(0.12, y_pos * 0.52, 0.20)
            side_strut.setHpr(0.0, 0.0, side_sign * 4.0)
            self._apply_material(side_strut, (0.18, 0.20, 0.24, 1.0), (0.06, 0.07, 0.08, 1.0), (0.32, 0.36, 0.40, 1.0), 20.0)

            suspension = root.attachNewNode(f"suspension-{side_name}")
            suspension.setPos(0.0, y_pos, 0.0)

            suspension_connector = self._build_box(f"suspension-connector-{side_name}", (0.22, 0.24, 0.28, 1.0))
            suspension_connector.reparentTo(suspension)
            suspension_connector.setScale(0.18, 0.08, 0.08)
            suspension_connector.setPos(0.66, 0.0, 0.12)
            self._apply_material(suspension_connector, (0.22, 0.24, 0.28, 1.0), (0.08, 0.09, 0.10, 1.0), (0.38, 0.42, 0.46, 1.0), 24.0)

            rocker = self._build_box(f"rocker-{side_name}", (0.20, 0.22, 0.26, 1.0))
            rocker.reparentTo(suspension)
            rocker.setScale(1.72, 0.07, 0.08)
            rocker.setPos(-0.10, 0.0, 0.10)
            rocker.setHpr(0, 0, side_sign * 4.0)
            self._apply_material(rocker, (0.18, 0.20, 0.22, 1.0), (0.08, 0.09, 0.10, 1.0), (0.34, 0.38, 0.42, 1.0), 24.0)

            front_leg = self._build_box(f"rocker-front-leg-{side_name}", (0.18, 0.20, 0.24, 1.0))
            front_leg.reparentTo(suspension)
            front_leg.setScale(0.10, 0.08, 0.36)
            front_leg.setPos(0.76, 0.0, -0.10)
            self._apply_material(front_leg, (0.18, 0.20, 0.24, 1.0), (0.06, 0.07, 0.08, 1.0), (0.32, 0.36, 0.40, 1.0), 20.0)

            middle_leg = self._build_box(f"rocker-middle-leg-{side_name}", (0.18, 0.20, 0.24, 1.0))
            middle_leg.reparentTo(suspension)
            middle_leg.setScale(0.10, 0.08, 0.34)
            middle_leg.setPos(0.00, 0.0, -0.14)
            self._apply_material(middle_leg, (0.18, 0.20, 0.24, 1.0), (0.06, 0.07, 0.08, 1.0), (0.32, 0.36, 0.40, 1.0), 20.0)

            rear_leg = self._build_box(f"rocker-rear-leg-{side_name}", (0.18, 0.20, 0.24, 1.0))
            rear_leg.reparentTo(suspension)
            rear_leg.setScale(0.10, 0.08, 0.36)
            rear_leg.setPos(-0.78, 0.0, -0.12)
            self._apply_material(rear_leg, (0.18, 0.20, 0.24, 1.0), (0.06, 0.07, 0.08, 1.0), (0.32, 0.36, 0.40, 1.0), 20.0)

            track_material = ((0.06, 0.07, 0.08, 1.0), (0.02, 0.02, 0.03, 1.0), (0.14, 0.16, 0.18, 1.0))
            track_top = self._build_box(f"track-top-{side_name}", (0.06, 0.07, 0.08, 1.0))
            track_top.reparentTo(suspension)
            track_top.setScale(1.92, 0.32, 0.12)
            track_top.setPos(0.0, 0.0, 0.40)
            self._apply_material(track_top, *track_material, 8.0)

            track_bottom = self._build_box(f"track-bottom-{side_name}", (0.06, 0.07, 0.08, 1.0))
            track_bottom.reparentTo(suspension)
            track_bottom.setScale(1.92, 0.32, 0.12)
            track_bottom.setPos(0.0, 0.0, -0.48)
            self._apply_material(track_bottom, *track_material, 8.0)

            for track_end, x_pos in (("front", 0.96), ("rear", -0.96)):
                end_plate = self._build_box(f"track-end-{side_name}-{track_end}", (0.06, 0.07, 0.08, 1.0))
                end_plate.reparentTo(suspension)
                end_plate.setScale(0.12, 0.32, 0.44)
                end_plate.setPos(x_pos, 0.0, -0.04)
                self._apply_material(end_plate, *track_material, 8.0)

            for plate_index, x_pos in enumerate(np.linspace(-0.78, 0.78, 7)):
                track_plate = self._build_box(f"track-plate-{side_name}-{plate_index}", (0.12, 0.13, 0.14, 1.0))
                track_plate.reparentTo(suspension)
                track_plate.setScale(0.09, 0.36, 0.055)
                track_plate.setPos(x_pos, 0.0, 0.47)
                track_plate.setHpr(0.0, 0.0, 90.0 if plate_index % 2 else 0.0)
                self._apply_material(track_plate, (0.12, 0.13, 0.14, 1.0), (0.04, 0.04, 0.05, 1.0), (0.22, 0.24, 0.26, 1.0), 10.0)
                self._tread_nodes.append((track_plate, float(x_pos), 0.0))

                lower_plate = self._build_box(f"track-lower-plate-{side_name}-{plate_index}", (0.10, 0.11, 0.12, 1.0))
                lower_plate.reparentTo(suspension)
                lower_plate.setScale(0.09, 0.36, 0.055)
                lower_plate.setPos(x_pos, 0.0, -0.55)
                lower_plate.setHpr(0.0, 0.0, 90.0 if plate_index % 2 else 0.0)
                self._apply_material(lower_plate, (0.10, 0.11, 0.12, 1.0), (0.03, 0.03, 0.04, 1.0), (0.18, 0.20, 0.22, 1.0), 8.0)
                self._tread_nodes.append((lower_plate, float(x_pos), 1.0))

            wheel_positions = [(0.82, "front"), (0.00, "middle"), (-0.82, "rear")]
            for x_pos, label in wheel_positions:
                wheel_outer = self._build_box(f"wheel-{side_name}-{label}", (0.10, 0.10, 0.12, 1.0))
                wheel_outer.reparentTo(suspension)
                wheel_outer.setScale(0.50, 0.30, 0.50)
                wheel_outer.setPos(x_pos, 0.0, -0.04)
                self._apply_material(wheel_outer, (0.12, 0.12, 0.14, 1.0), (0.04, 0.04, 0.05, 1.0), (0.32, 0.32, 0.34, 1.0), 16.0)

                wheel_inner = self._build_box(f"wheel-hub-{side_name}-{label}", (0.42, 0.44, 0.48, 1.0))
                wheel_inner.reparentTo(suspension)
                wheel_inner.setScale(0.20, 0.13, 0.20)
                wheel_inner.setPos(x_pos, 0.0, 0.08)
                self._apply_material(wheel_inner, (0.48, 0.50, 0.54, 1.0), (0.18, 0.18, 0.20, 1.0), (0.80, 0.82, 0.85, 1.0), 80.0)

                wheel_cap = self._build_box(f"wheel-cap-{side_name}-{label}", (0.92, 0.86, 0.58, 1.0))
                wheel_cap.reparentTo(suspension)
                wheel_cap.setScale(0.08, 0.08, 0.08)
                wheel_cap.setPos(x_pos, 0.0, 0.11)
                self._apply_material(wheel_cap, (0.98, 0.88, 0.54, 1.0), (0.30, 0.24, 0.10, 1.0), (1.0, 0.96, 0.78, 1.0), 120.0)

                wheel_tread = self._build_box(f"wheel-tread-{side_name}-{label}", (0.08, 0.08, 0.09, 1.0))
                wheel_tread.reparentTo(suspension)
                wheel_tread.setScale(0.56, 0.08, 0.16)
                wheel_tread.setPos(x_pos, 0.0, -0.02)
                self._apply_material(wheel_tread, (0.08, 0.08, 0.09, 1.0), (0.03, 0.03, 0.03, 1.0), (0.16, 0.16, 0.18, 1.0), 6.0)

        self._drill_stage_np = root.attachNewNode("drill-stage")
        self._drill_stage_np.setPos(0.0, 0.0, 0.0)

        drill_base = self._build_box("drill-base", (0.28, 0.30, 0.34, 1.0))
        drill_base.reparentTo(self._drill_stage_np)
        drill_base.setScale(0.16, 0.14, 0.22)
        drill_base.setPos(0.90, 0.40, 0.02)
        drill_base.setHpr(10.0, -8.0, 0.0)
        self._apply_material(drill_base, (0.26, 0.28, 0.32, 1.0), (0.10, 0.11, 0.12, 1.0), (0.48, 0.52, 0.56, 1.0), 34.0)

        drill_shoulder = self._build_box("drill-shoulder", (0.34, 0.36, 0.40, 1.0))
        drill_shoulder.reparentTo(self._drill_stage_np)
        drill_shoulder.setScale(0.10, 0.10, 0.18)
        drill_shoulder.setPos(1.00, 0.42, -0.08)
        drill_shoulder.setHpr(18.0, 0.0, -12.0)
        self._apply_material(drill_shoulder, (0.32, 0.34, 0.38, 1.0), (0.12, 0.13, 0.14, 1.0), (0.56, 0.60, 0.64, 1.0), 40.0)

        drill_forearm = self._build_box("drill-forearm", (0.34, 0.36, 0.40, 1.0))
        drill_forearm.reparentTo(self._drill_stage_np)
        drill_forearm.setScale(0.09, 0.07, 0.54)
        drill_forearm.setPos(1.14, 0.42, -0.26)
        drill_forearm.setHpr(8.0, 6.0, 0.0)
        self._apply_material(drill_forearm, (0.32, 0.34, 0.38, 1.0), (0.12, 0.13, 0.14, 1.0), (0.56, 0.60, 0.64, 1.0), 40.0)

        drill_joint = self._build_box("drill-joint", (0.92, 0.78, 0.30, 1.0))
        drill_joint.reparentTo(self._drill_stage_np)
        drill_joint.setScale(0.10, 0.10, 0.10)
        drill_joint.setPos(1.24, 0.42, -0.56)
        self._apply_material(drill_joint, (0.94, 0.80, 0.30, 1.0), (0.22, 0.18, 0.07, 1.0), (1.0, 0.92, 0.48, 1.0), 86.0)

        drill = self._build_box("drill", (0.90, 0.74, 0.20, 1.0))
        drill.reparentTo(self._drill_stage_np)
        drill.setScale(0.07, 0.07, 0.72)
        drill.setHpr(90, 0, 0)
        drill.setPos(1.24, 0.42, -0.86)
        self._apply_material(drill, (0.95, 0.80, 0.22, 1.0), (0.22, 0.16, 0.05, 1.0), (1.0, 0.90, 0.46, 1.0), 92.0)

        drill_tip = self._build_box("drill-tip", (1.0, 0.92, 0.50, 1.0))
        drill_tip.reparentTo(self._drill_stage_np)
        drill_tip.setScale(0.04, 0.04, 0.10)
        drill_tip.setHpr(90, 0, 0)
        drill_tip.setPos(1.24, 0.42, -1.06)
        self._apply_material(drill_tip, (1.0, 0.92, 0.58, 1.0), (0.30, 0.24, 0.08, 1.0), (1.0, 0.98, 0.70, 1.0), 120.0)

        drill_guard = self._build_box("drill-guard", (0.18, 0.20, 0.24, 1.0))
        drill_guard.reparentTo(self._drill_stage_np)
        drill_guard.setScale(0.16, 0.09, 0.12)
        drill_guard.setPos(1.08, 0.42, -0.68)
        self._apply_material(drill_guard, (0.18, 0.20, 0.24, 1.0), (0.06, 0.07, 0.08, 1.0), (0.34, 0.38, 0.42, 1.0), 20.0)

        self._measure_stage_np = root.attachNewNode("measure-stage")
        self._measure_stage_np.setPos(0.0, 0.0, 0.0)

        antenna = self._build_box("antenna", (0.80, 0.82, 0.86, 1.0))
        antenna.reparentTo(root)
        antenna.setScale(0.04, 0.04, 0.70)
        antenna.setPos(-0.04, -0.34, 1.86)
        self._apply_material(antenna, (0.84, 0.86, 0.90, 1.0), (0.28, 0.28, 0.30, 1.0), (0.60, 0.64, 0.68, 1.0), 42.0)

        antenna_tip = self._build_box("antenna-tip", (0.18, 0.94, 0.82, 1.0))
        antenna_tip.reparentTo(root)
        antenna_tip.setScale(0.10, 0.10, 0.10)
        antenna_tip.setPos(-0.04, -0.34, 2.34)
        self._apply_material(antenna_tip, (0.18, 0.94, 0.82, 1.0), (0.04, 0.18, 0.16, 1.0), (0.60, 1.0, 0.92, 1.0), 72.0)

        sensor_box = self._build_box("sensor-box", (0.24, 0.28, 0.32, 1.0))
        sensor_box.reparentTo(self._measure_stage_np)
        sensor_box.setScale(0.26, 0.18, 0.12)
        sensor_box.setPos(-0.82, -0.14, 0.46)
        self._apply_material(sensor_box, (0.24, 0.28, 0.32, 1.0), (0.08, 0.10, 0.12, 1.0), (0.44, 0.48, 0.52, 1.0), 36.0)

        sensor_lid = self._build_box("sensor-lid", (0.34, 0.38, 0.42, 1.0))
        sensor_lid.reparentTo(self._measure_stage_np)
        sensor_lid.setScale(0.26, 0.18, 0.05)
        sensor_lid.setPos(-0.82, -0.14, 0.58)
        self._apply_material(sensor_lid, (0.34, 0.38, 0.42, 1.0), (0.10, 0.12, 0.14, 1.0), (0.56, 0.60, 0.66, 1.0), 54.0)

        support_arm = self._build_box("support-arm", (0.18, 0.20, 0.24, 1.0))
        support_arm.reparentTo(self._measure_stage_np)
        support_arm.setScale(0.08, 0.26, 0.08)
        support_arm.setPos(0.60, -0.38, 0.48)
        support_arm.setHpr(-12.0, 0.0, 20.0)
        self._apply_material(support_arm, (0.18, 0.20, 0.24, 1.0), (0.06, 0.07, 0.08, 1.0), (0.32, 0.36, 0.40, 1.0), 20.0)

        self._shadow_np = self._make_rover_shadow()

        return root

    def _build_scene(self) -> None:
        self._setup_lights()
        self._sky_np = self._make_sky_dome()
        self.terrain_np = self._make_terrain()
        self.rover_np = self._build_rover()
        self.waypoint_nodes: List[NodePath] = []

        for index, y in enumerate(np.linspace(-21.0, 21.0, 17)):
            row = self._build_box(f"field-row-{index}", (0.44, 0.35, 0.18, 1.0))
            row.reparentTo(self.render)
            row.setScale(43.0, 0.18, 0.10)
            row.setPos(0.0, y, self.terrain.height(0.0, y) + 0.09)
            row.setColor((0.48 + 0.02 * index, 0.43 + 0.01 * index, 0.28, 1.0))
            self._field_rows.append(row)

            for tuft_index, x in enumerate(np.linspace(-20.0, 20.0, 9)):
                tuft = self._build_box(f"grass-tuft-{index}-{tuft_index}", (0.18, 0.42, 0.10, 1.0))
                tuft.reparentTo(self.render)
                tuft.setScale(0.045, 0.045, 0.22 + 0.04 * ((index + tuft_index) % 3))
                tuft.setPos(x + (0.8 if index % 2 else -0.8), y + 0.34, self.terrain.height(x, y + 0.34) + 0.16)
                tuft.setHpr(-10.0 + (tuft_index % 3) * 8.0, 0.0, -18.0 + index * 3.0)
                tuft.setColor((0.18 + 0.02 * (index % 3), 0.42 + 0.03 * (tuft_index % 3), 0.10, 1.0))
                self._grass_nodes.append(tuft)

        for n in range(16):
            fence_post = self._build_box(f"fence-post-{n}", (0.62, 0.52, 0.34, 1.0))
            fence_post.reparentTo(self.render)
            fence_post.setScale(0.12, 0.12, 1.0)
            x = -22.5 + n * 3.0
            fence_post.setPos(x, -22.4, self.terrain.height(x, -22.4) + 0.55)

        for n in range(16):
            fence_post = self._build_box(f"fence-post-2-{n}", (0.62, 0.52, 0.34, 1.0))
            fence_post.reparentTo(self.render)
            fence_post.setScale(0.12, 0.12, 1.0)
            y = -22.0 + n * 3.0
            fence_post.setPos(-22.4, y, self.terrain.height(-22.4, y) + 0.55)

        weather = weather_profile(self.terrain_mode)
        sun_texture = self._make_soft_disc_texture(
            "sun",
            (weather["sun_glow"][0], weather["sun_glow"][1], weather["sun_glow"][2]),
            (weather["sun_glow"][0], weather["sun_glow"][1], weather["sun_glow"][2], 0.0),
        )
        self._sun_np = self._make_billboard_sprite(
            "sun-disc",
            sun_texture,
            22.0,
            (22.0, 30.0, 82.0),
            (1.0, 0.94, 0.76, 0.80),
        )

        cloud_texture = self._make_soft_disc_texture("cloud", (0.98, 0.98, 1.0), (0.96, 0.98, 1.0, 0.0))
        for index, cloud_pos in enumerate([(-35.0, 18.0, 70.0), (10.0, 42.0, 76.0), (38.0, -4.0, 68.0)]):
            cloud = self._make_billboard_sprite(
                f"cloud-{index}",
                cloud_texture,
                18.0 + index * 3.5,
                cloud_pos,
                (1.0, 1.0, 1.0, weather["cloud_alpha"]),
            )
            cloud.setDepthWrite(False)
            self._cloud_nodes.append(cloud)

        for idx, waypoint in enumerate(self.waypoints):
            marker = self._build_waypoint_marker(self._waypoint_colour(idx))
            marker.reparentTo(self.render)
            marker.setPos(waypoint.x, waypoint.y, self.terrain.height(waypoint.x, waypoint.y) + 0.45)
            self.waypoint_nodes.append(marker)

        if len(self.waypoints) >= 2:
            route = LineSegs("mission-route")
            route.setThickness(2.5)
            route.setColor(0.18, 0.92, 0.72, 0.72)
            first = self.waypoints[0]
            route.moveTo(first.x, first.y, self.terrain.height(first.x, first.y) + 0.16)
            for waypoint in self.waypoints[1:]:
                route.drawTo(waypoint.x, waypoint.y, self.terrain.height(waypoint.x, waypoint.y) + 0.16)
            self._route_np = self.render.attachNewNode(route.create())

        self._spawn_field_obstacles()

    def _spawn_field_obstacles(self) -> None:
        obstacle_specs = [
            ("tree-1", "tree", -4.5, -0.5, 0.95),
            ("tree-2", "tree", 2.8, 4.0, 1.05),
            ("tree-3", "tree", 6.8, -3.8, 0.90),
            ("tree-4", "tree", -18.0, -15.0, 1.25),
            ("tree-5", "tree", -14.0, 14.0, 1.10),
            ("tree-6", "tree", -2.0, 18.0, 1.35),
            ("tree-7", "tree", 11.0, 17.0, 1.20),
            ("tree-8", "tree", 19.0, 10.0, 1.30),
            ("tree-9", "tree", 18.0, -12.0, 1.15),
            ("tree-10", "tree", -18.0, 7.0, 1.05),
            ("tree-11", "tree", 12.0, -20.0, 1.30),
            ("rock-1", "rock", -1.2, 2.8, 0.55),
            ("rock-2", "rock", 4.9, -1.6, 0.45),
            ("rock-3", "rock", -6.2, 5.2, 0.60),
        ]
        self._obstacle_nodes.clear()
        self._detected_obstacles.clear()

        for name, kind, x, y, scale in obstacle_specs:
            if kind == "tree":
                obstacle = self._build_tree(name, scale)
            else:
                obstacle = self._build_rock(name, scale)
            obstacle.reparentTo(self.render)
            obstacle.setPos(x, y, self.terrain.height(x, y))
            self._obstacle_nodes.append((name, obstacle, x, y, scale))

    def _build_tree(self, name: str, scale: float) -> NodePath:
        root = self.render.attachNewNode(name)

        trunk = self._build_box(f"{name}-trunk", (0.42, 0.28, 0.12, 1.0))
        trunk.reparentTo(root)
        trunk.setScale(0.14 * scale, 0.14 * scale, 0.72 * scale)
        trunk.setPos(0.0, 0.0, 0.32 * scale)
        trunk.setHpr(-3.0, 0.0, 4.0)
        self._apply_material(trunk, (0.40, 0.26, 0.12, 1.0), (0.16, 0.10, 0.05, 1.0), (0.52, 0.34, 0.18, 1.0), 10.0)

        for side, x, z, rotation in (("left", -0.22, 0.78, -18.0), ("right", 0.20, 0.82, 20.0)):
            branch = self._build_box(f"{name}-branch-{side}", (0.30, 0.20, 0.10, 1.0))
            branch.reparentTo(root)
            branch.setScale(0.40 * scale, 0.07 * scale, 0.07 * scale)
            branch.setPos(x * scale, 0.0, z * scale)
            branch.setHpr(0.0, 0.0, rotation)
            self._apply_material(branch, (0.34, 0.20, 0.08, 1.0), (0.12, 0.07, 0.03, 1.0), (0.44, 0.26, 0.10, 1.0), 8.0)

        foliage = [
            ("lower", 0.42, 0.34, 0.25, -0.02, 0.0, 0.90, (0.12, 0.34, 0.10, 1.0)),
            ("left", 0.30, 0.28, 0.25, -0.28, 0.02, 1.08, (0.16, 0.46, 0.14, 1.0)),
            ("right", 0.31, 0.27, 0.24, 0.27, -0.04, 1.10, (0.20, 0.52, 0.16, 1.0)),
            ("top", 0.25, 0.24, 0.24, 0.02, 0.04, 1.30, (0.26, 0.58, 0.18, 1.0)),
        ]
        for part, sx, sy, sz, x, y, z, diffuse in foliage:
            canopy = self._build_box(f"{name}-canopy-{part}", diffuse)
            canopy.reparentTo(root)
            canopy.setScale(sx * scale, sy * scale, sz * scale)
            canopy.setPos(x * scale, y * scale, z * scale)
            canopy.setHpr((x + y) * 18.0, y * 12.0, x * 20.0)
            self._apply_material(canopy, diffuse, (diffuse[0] * 0.42, diffuse[1] * 0.42, diffuse[2] * 0.42, 1.0), (diffuse[0] * 1.3, diffuse[1] * 1.3, diffuse[2] * 1.3, 1.0), 20.0)

        return root

    def _build_rock(self, name: str, scale: float) -> NodePath:
        root = self.render.attachNewNode(name)

        base = self._build_box(f"{name}-base", (0.44, 0.42, 0.38, 1.0))
        base.reparentTo(root)
        base.setScale(0.30 * scale, 0.24 * scale, 0.16 * scale)
        base.setHpr(8.0, -8.0, 14.0)
        base.setPos(0.0, 0.0, 0.08 * scale)
        self._apply_material(base, (0.40, 0.38, 0.34, 1.0), (0.12, 0.12, 0.12, 1.0), (0.60, 0.58, 0.54, 1.0), 24.0)

        cap = self._build_box(f"{name}-cap", (0.54, 0.52, 0.46, 1.0))
        cap.reparentTo(root)
        cap.setScale(0.18 * scale, 0.16 * scale, 0.12 * scale)
        cap.setHpr(-12.0, 10.0, -18.0)
        cap.setPos(0.12 * scale, -0.04 * scale, 0.20 * scale)
        self._apply_material(cap, (0.52, 0.50, 0.44, 1.0), (0.16, 0.16, 0.14, 1.0), (0.72, 0.70, 0.64, 1.0), 30.0)

        chip = self._build_box(f"{name}-chip", (0.50, 0.48, 0.42, 1.0))
        chip.reparentTo(root)
        chip.setScale(0.12 * scale, 0.10 * scale, 0.08 * scale)
        chip.setHpr(18.0, 0.0, 24.0)
        chip.setPos(-0.14 * scale, 0.10 * scale, 0.16 * scale)
        self._apply_material(chip, (0.48, 0.46, 0.40, 1.0), (0.14, 0.14, 0.12, 1.0), (0.68, 0.66, 0.60, 1.0), 28.0)

        return root

    def _setup_lights(self) -> None:
        self.render.setShaderAuto()
        profile = lighting_profile(self.terrain_mode)

        fog = Fog("atmosphere")
        fog.setColor(profile["fog_color"][0], profile["fog_color"][1], profile["fog_color"][2])
        fog.setExpDensity(profile["fog_density"])
        self.render.setFog(fog)

        ambient = AmbientLight("ambient")
        ambient.setColor(profile["ambient"])
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        key_light = DirectionalLight("sun")
        key_light.setColor(profile["sun_color"])
        key_np = self.render.attachNewNode(key_light)
        key_np.setHpr(-35, -62, 0)
        self.render.setLight(key_np)

        fill_light = DirectionalLight("fill")
        fill_light.setColor(profile["fill_color"])
        fill_np = self.render.attachNewNode(fill_light)
        fill_np.setHpr(145, -26, 0)
        self.render.setLight(fill_np)

        self.setBackgroundColor(*profile["background"])

    def _build_hud(self) -> None:
        self.hud = OnscreenText(
            text="",
            parent=self.a2dTopLeft,
            align=TextNode.ALeft,
            pos=(0.05, -0.07),
            scale=0.05,
            fg=(0.98, 0.98, 0.98, 1.0),
            mayChange=True,
            shadow=(0.05, 0.05, 0.05, 0.9),
        )
        self.help = OnscreenText(
            text="W/S throttle   A/D steer   M manual   C camera   Space pause   R reset   X zero input   1 auto   Q quit",
            parent=self.a2dBottomLeft,
            align=TextNode.ALeft,
            pos=(0.05, 0.06),
            scale=0.045,
            fg=(0.94, 0.94, 0.94, 1.0),
            mayChange=False,
            shadow=(0.05, 0.05, 0.05, 0.9),
        )

    def _waypoint_colour(self, idx: int) -> Tuple[float, float, float, float]:
        return (0.86, 0.90, 0.34, 1.0) if idx == self.target_idx else (0.62, 0.78, 0.24, 1.0)

    def _current_waypoint(self) -> Optional[Waypoint3D]:
        if self.target_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.target_idx]

    def _scan_obstacles(self) -> None:
        nearest_name = None
        nearest_distance = float("inf")

        for name, obstacle_np, x, y, _scale in self._obstacle_nodes:
            dx = x - self.state.x
            dy = y - self.state.y
            distance = math.hypot(dx, dy)
            bearing = _wrap_angle_deg(math.degrees(math.atan2(dy, dx)) - self.state.heading_deg)
            visible = distance <= 11.0 and abs(bearing) <= 72.0

            if visible and distance < nearest_distance:
                nearest_name = name
                nearest_distance = distance

            if visible:
                self._detected_obstacles.add(name)
                obstacle_np.setColorScale(1.0, 1.0, 0.82, 1.0)
            else:
                obstacle_np.setColorScale(1.0, 1.0, 1.0, 1.0)

        if nearest_name is None:
            self._nearest_obstacle_text = "none"
        else:
            self._nearest_obstacle_text = f"{nearest_name} @ {nearest_distance:.1f} m"

    def toggle_manual_mode(self) -> None:
        self.manual_mode = not self.manual_mode

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def zero_manual_input(self) -> None:
        self.manual_throttle = 0.0
        self.manual_steer = 0.0

    def reset_pose(self) -> None:
        self.state = RoverState(x=-8.0, y=-7.0, heading_deg=25.0, speed=0.0, z=self.terrain.height(-8.0, -7.0))
        self.target_idx = 0
        self.step_idx = 0
        self.finished = False
        self._trail_points.clear()
        self._trail_dirty = True
        self.manual_throttle = 0.0
        self.manual_steer = 0.0
        self.mission_phase = "drive_to_drill"
        self.mission_hold_timer = 0.0
        self._drill_lift = 0.0
        self._measure_lift = 0.0
        self._drill_spin = 0.0
        self._mission_measure_target = None
        self._drill_cycle_complete = False
        self._measurement_complete = False
        self._nearest_obstacle_text = "none"
        self._trail_rebuild()

    def capture_snapshot(self, path: Optional[str] = None, exit_after_save: bool = False) -> bool:
        output_path = path or self._snapshot_path or "rover_snapshot.png"
        if self.win is None:
            log.warning("Snapshot skipped because no window is available")
            return False
        try:
            saved = self.win.saveScreenshot(Filename(output_path))
        except Exception:
            log.exception("Failed to save snapshot -> %s", output_path)
            return False
        if saved:
            log.info("Snapshot saved -> %s", output_path)
            self._snapshot_taken = True
            if exit_after_save:
                self._snapshot_exit = True
        else:
            log.warning("Snapshot capture returned no file -> %s", output_path)
        return bool(saved)

    def _estimate_path_curvature(self, target: Optional[Waypoint3D], heading_error: float, distance: float) -> float:
        if target is None or distance < 1e-6:
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

    def _drive_towards_point(self, target_x: float, target_y: float, dt: float, tolerance: Optional[float] = None) -> bool:
        tolerance_m = self.waypoint_tolerance_m if tolerance is None else tolerance
        dx = target_x - self.state.x
        dy = target_y - self.state.y
        distance = math.hypot(dx, dy)
        desired_heading = math.degrees(math.atan2(dy, dx))
        heading_error = _wrap_angle_deg(desired_heading - self.state.heading_deg)
        slope_x, slope_y = self.terrain.gradient(self.state.x, self.state.y)
        slope_mag = math.hypot(slope_x, slope_y)

        steering = _clamp(heading_error * self.steering_gain, -90.0, 90.0)
        self.state.heading_deg = _wrap_angle_deg(self.state.heading_deg + steering * dt)

        slope_penalty = 1.0 - _clamp(slope_mag * 0.18, 0.0, 0.22)
        target_speed = self.max_speed_mps * _clamp(distance / 2.5, 0.25, 1.0) * slope_penalty
        if distance <= tolerance_m:
            target_speed = 0.0

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
        self._trail_points.append((self.state.x, self.state.y, self.state.z))
        self._trail_dirty = True

        return distance <= tolerance_m

    def _limit_to_world(self) -> None:
        half = self.terrain.span_m / 2.0
        self.state.x = _clamp(self.state.x, -half, half)
        self.state.y = _clamp(self.state.y, -half, half)

    def _update_body_orientation(self) -> None:
        slope_x, slope_y = self.terrain.gradient(self.state.x, self.state.y)
        self.state.pitch_deg = math.degrees(math.atan(-slope_x))
        self.state.roll_deg = math.degrees(math.atan(slope_y))

    def _step_manual(self, dt: float) -> None:
        throttle_target = 0.0
        steer_target = 0.0
        if self._keys["w"]:
            throttle_target += 1.0
        if self._keys["s"]:
            throttle_target -= 1.0
        if self._keys["a"]:
            steer_target += 1.0
        if self._keys["d"]:
            steer_target -= 1.0

        self.manual_throttle = _lerp(self.manual_throttle, throttle_target, dt * 4.5)
        self.manual_steer = _lerp(self.manual_steer, steer_target, dt * 4.5)

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
        self._trail_points.append((self.state.x, self.state.y, self.state.z))
        self._trail_dirty = True

    def _step_autonomous(self, dt: float) -> bool:
        target = self._current_waypoint()
        if target is None:
            self.state.speed = 0.0
            self._update_body_orientation()
            self._trail_points.append((self.state.x, self.state.y, self.state.z))
            self._trail_dirty = True
            return True

        if self._drive_towards_point(target.x, target.y, dt):
            log.info("Reached waypoint %s", target.name)
            self.target_idx += 1
            self._refresh_waypoints()
        return self._current_waypoint() is None

    def _step_mission(self, dt: float) -> bool:
        target = self._current_waypoint()
        if target is None:
            self.state.speed = 0.0
            return True

        if self.mission_phase == "drive_to_drill":
            if self._drive_towards_point(target.x, target.y, dt):
                self.state.speed = 0.0
                self.mission_phase = "lower_drill"
                self.mission_hold_timer = 0.0
                self._drill_cycle_complete = False
                self._measurement_complete = False

        elif self.mission_phase == "lower_drill":
            self._drill_lift = _clamp(self._drill_lift + dt * 0.9, 0.0, 1.0)
            if self._drill_lift >= 1.0:
                self.mission_phase = "drill_spin"
                self.mission_hold_timer = 0.0

        elif self.mission_phase == "drill_spin":
            self._drill_spin += dt * 12.0
            self.mission_hold_timer += dt
            if self.mission_hold_timer >= 2.0:
                self._drill_cycle_complete = True
                self.mission_phase = "raise_drill"

        elif self.mission_phase == "raise_drill":
            self._drill_lift = _clamp(self._drill_lift - dt * 0.9, 0.0, 1.0)
            if self._drill_lift <= 0.0:
                heading_rad = math.radians(self.state.heading_deg)
                self._mission_measure_target = (
                    self.state.x + math.cos(heading_rad) * 0.45,
                    self.state.y + math.sin(heading_rad) * 0.45,
                )
                self.mission_phase = "drive_to_measure"

        elif self.mission_phase == "drive_to_measure":
            target_x, target_y = self._mission_measure_target or (self.state.x, self.state.y)
            if self._drive_towards_point(target_x, target_y, dt, tolerance=0.20):
                self.state.speed = 0.0
                self.mission_phase = "lower_measure"
                self.mission_hold_timer = 0.0

        elif self.mission_phase == "lower_measure":
            self._measure_lift = _clamp(self._measure_lift + dt * 1.0, 0.0, 1.0)
            if self._measure_lift >= 1.0:
                self.mission_phase = "measure_hold"
                self.mission_hold_timer = 0.0

        elif self.mission_phase == "measure_hold":
            self.mission_hold_timer += dt
            if self.mission_hold_timer >= 1.1:
                self._measurement_complete = True
                log.info("Measurement complete at %s", target.name)
                self.mission_phase = "raise_measure"

        elif self.mission_phase == "raise_measure":
            self._measure_lift = _clamp(self._measure_lift - dt * 1.0, 0.0, 1.0)
            if self._measure_lift <= 0.0:
                self.target_idx += 1
                self._refresh_waypoints()
                self.mission_phase = "drive_to_drill"
                self._mission_measure_target = None
                self.mission_hold_timer = 0.0
                self._drill_lift = 0.0
                self._measure_lift = 0.0

        return self._current_waypoint() is None

    def _refresh_waypoints(self) -> None:
        for idx, node in enumerate(self.waypoint_nodes):
            node.setColor(*self._waypoint_colour(idx))
        if self._current_waypoint() is not None and self.target_idx < len(self.waypoint_nodes):
            self.waypoint_nodes[self.target_idx].setColor(0.0, 0.85, 0.65, 1.0)

    def _sync_scene(self) -> None:
        self.rover_np.setPos(self.state.x, self.state.y, self.state.z)
        self.rover_np.setH(self.state.heading_deg)
        self.rover_np.setP(self.state.pitch_deg)
        self.rover_np.setR(self.state.roll_deg)

        cam_heading = math.radians(self.state.heading_deg)
        if self._camera_mode == "overhead":
            desired_eye = LVector3f(self.state.x, self.state.y - 0.5, self.state.z + 22.0)
            desired_target = LVector3f(self.state.x, self.state.y, self.state.z)
        elif self._camera_mode == "cockpit":
            desired_eye = LVector3f(
                self.state.x + math.cos(cam_heading) * 0.65,
                self.state.y + math.sin(cam_heading) * 0.65,
                self.state.z + 1.65,
            )
            desired_target = LVector3f(
                self.state.x + math.cos(cam_heading) * 7.0,
                self.state.y + math.sin(cam_heading) * 7.0,
                self.state.z + 1.15,
            )
        else:
            camera_distance = 10.0
            desired_eye = LVector3f(
                self.state.x - math.cos(cam_heading) * camera_distance + math.sin(cam_heading) * 3.0,
                self.state.y - math.sin(cam_heading) * camera_distance - math.cos(cam_heading) * 3.0,
                self.state.z + 8.5,
            )
            desired_target = LVector3f(
                self.state.x + math.cos(cam_heading) * 1.5,
                self.state.y + math.sin(cam_heading) * 1.5,
                self.state.z + 0.5,
            )
        self.camLens.setFov(64.0)
        alpha = 0.12 if self._render_frame else 1.0
        self._camera_pos = LVector3f(
            _lerp(self._camera_pos.x, desired_eye.x, alpha),
            _lerp(self._camera_pos.y, desired_eye.y, alpha),
            _lerp(self._camera_pos.z, desired_eye.z, alpha),
        )
        self._camera_target = LVector3f(
            _lerp(self._camera_target.x, desired_target.x, alpha),
            _lerp(self._camera_target.y, desired_target.y, alpha),
            _lerp(self._camera_target.z, desired_target.z, alpha),
        )
        self.camera.setPos(self._camera_pos)
        self.camera.lookAt(self._camera_target)

        tread_phase = (self._wheel_rotation * 0.08) % 1.8
        for tread, base_x, direction in self._tread_nodes:
            offset = ((tread_phase * (-1.0 if direction else 1.0) + base_x + 0.9) % 1.8) - 0.9
            tread.setX(offset)

        for index, tuft in enumerate(self._grass_nodes):
            tuft.setR(-18.0 + 7.0 * math.sin(self._render_frame * 0.025 + index * 0.7))

        if self._sky_np is not None:
            self._sky_np.setPos(self.state.x, self.state.y, self.state.z - 6.0)
        if self._sun_np is not None:
            self._sun_np.setPos(self.state.x + 26.0, self.state.y + 34.0, self.state.z + 72.0)
        cloud_drift = self._render_frame * 0.012
        for idx, cloud in enumerate(self._cloud_nodes):
            cloud.setPos(
                self.state.x + (-24.0 + idx * 18.0) + cloud_drift,
                self.state.y + (20.0 - idx * 8.0),
                self.state.z + (24.0 + idx * 3.0),
            )
        if self._shadow_np is not None:
            self._shadow_np.setPos(self.state.x + 0.12, self.state.y - 0.06, self.terrain.height(self.state.x, self.state.y) + 0.012)
            self._shadow_np.setScale(1.2 + self.state.speed * 0.18, 1.0 + self.state.speed * 0.10, 1.0)

        if self._drill_stage_np is not None:
            self._drill_stage_np.setPos(0.0, 0.0, -0.58 * self._drill_lift)
            self._drill_stage_np.setP(-3.0 - 5.0 * self._drill_lift)
            self._drill_stage_np.setHpr(self._drill_stage_np.getH(), self._drill_stage_np.getP(), 0.0)
        if self._measure_stage_np is not None:
            self._measure_stage_np.setPos(0.0, 0.0, -0.48 * self._measure_lift)
            self._measure_stage_np.setHpr(0.0, 0.0, 0.0)

        if self.mission_mode:
            # Keep the drill visibly turning during the drill phase.
            if self.mission_phase in ("drill_spin", "lower_drill", "raise_drill"):
                spin = self._drill_spin
            else:
                spin = 0.0
            for node_name in ["drill", "drill-tip", "drill-joint", "drill-guard"]:
                node = self.rover_np.find(f"**/{node_name}")
                if not node.isEmpty():
                    node.setR(spin)
            if self._measurement_complete:
                self.rover_np.find("**/sensor-box").setR(0.0)
                self.rover_np.find("**/sensor-lid").setR(0.0)

            self._scan_obstacles()

        self._update_trail()
        self._update_hud()
        self._render_frame += 1

    def _update_hud(self) -> None:
        target = self._current_waypoint()
        self.hud.setText(
            f"Target: {target.name if target else 'none'}\n"
            f"Pose: x={self.state.x:+.2f} m  y={self.state.y:+.2f} m  z={self.state.z:+.2f} m\n"
            f"Heading: {self.state.heading_deg:+.1f} deg  Speed: {self.state.speed:.2f} m/s\n"
            f"Mode: {'manual' if self.manual_mode else self.control_mode}  Terrain: {self.terrain_mode}\n"
            f"Obstacle: {self._nearest_obstacle_text}  Detected: {len(self._detected_obstacles)}\n"
            f"Mission: {'on' if self.mission_mode else 'off'}  Phase: {self.mission_phase}  Route: {min(self.target_idx, len(self.waypoints))}/{len(self.waypoints)}\n"
            f"Camera: {self._camera_mode}  Tread speed: {self._wheel_rotation % 360.0:.0f} deg\n"
            f"Drill lift: {self._drill_lift:.2f}  Measure lift: {self._measure_lift:.2f}  pitch {self.state.pitch_deg:+.1f} deg  roll {self.state.roll_deg:+.1f} deg"
        )

    def _trail_rebuild(self) -> None:
        if self._trail_np is not None:
            self._trail_np.removeNode()
            self._trail_np = None

    def _update_trail(self) -> None:
        # Only rebuild the LineSegs geometry when new points actually came
        # in (or reset_pose asked for a rebuild). Previously this ran a
        # full remove+recreate of the trail geometry every frame - including
        # every paused frame, where the rover isn't moving and the trail
        # never changes.
        if not self._trail_dirty:
            return

        if len(self._trail_points) < 2:
            self._trail_rebuild()
            self._trail_dirty = False
            return

        self._trail_rebuild()
        left_track = LineSegs("left-tire-track")
        right_track = LineSegs("right-tire-track")
        left_track.setThickness(3.0)
        right_track.setThickness(3.0)
        left_track.setColor(0.16, 0.10, 0.05, 0.75)
        right_track.setColor(0.16, 0.10, 0.05, 0.75)
        points = list(self._trail_points)
        offsets = []
        for point_index, point in enumerate(points):
            reference = points[min(point_index + 1, len(points) - 1)]
            if reference == point and point_index > 0:
                reference = points[point_index - 1]
            tangent = math.atan2(reference[1] - point[1], reference[0] - point[0])
            side_x = -math.sin(tangent) * 0.34
            side_y = math.cos(tangent) * 0.34
            offsets.append((side_x, side_y))

        first = points[0]
        first_offset = offsets[0]
        left_track.moveTo(first[0] + first_offset[0], first[1] + first_offset[1], first[2] + 0.07)
        right_track.moveTo(first[0] - first_offset[0], first[1] - first_offset[1], first[2] + 0.07)
        for point, (side_x, side_y) in zip(points[1:], offsets[1:]):
            left_track.drawTo(point[0] + side_x, point[1] + side_y, point[2] + 0.07)
            right_track.drawTo(point[0] - side_x, point[1] - side_y, point[2] + 0.07)
        trail_root = self.render.attachNewNode("tire-tracks")
        self._trail_np = trail_root
        left_track_node = trail_root.attachNewNode(left_track.create())
        right_track_node = trail_root.attachNewNode(right_track.create())
        left_track_node.setDepthOffset(2)
        right_track_node.setDepthOffset(2)
        self._trail_dirty = False

    def _update_task(self, task):
        if self._quit_requested:
            self.finished = True
            self.userExit()
            return task.done

        if self.finished:
            return task.done

        dt = _clamp(ClockObject.getGlobalClock().getDt(), 0.008, 0.05)

        if self.paused:
            self.state.speed = 0.0
            self._update_body_orientation()
            self._sync_scene()
            return task.cont

        if self.manual_mode:
            self._step_manual(dt)
        elif self.mission_mode:
            self.finished = self._step_mission(dt)
        else:
            self.finished = self._step_autonomous(dt)

        target = self._current_waypoint()
        target_name = target.name if target else "done"
        dx = (target.x - self.state.x) if target else 0.0
        dy = (target.y - self.state.y) if target else 0.0
        target_distance_m = math.hypot(dx, dy) if target else 0.0
        target_bearing_deg = _wrap_angle_deg(math.degrees(math.atan2(dy, dx)) - self.state.heading_deg) if target else 0.0
        slope_x, slope_y = self.terrain.gradient(self.state.x, self.state.y)
        slope_mag = math.hypot(slope_x, slope_y)
        path_curvature = self._estimate_path_curvature(target, target_bearing_deg, target_distance_m)
        power_draw = self._estimate_power_draw(target_bearing_deg, slope_mag, self.state.speed)
        wheel_slip = self._estimate_wheel_slip(slope_mag, target_bearing_deg)

        terrain_height = self.terrain.height(self.state.x, self.state.y)
        self.log.record(
            self.step_idx,
            self.run_label,
            self.control_mode,
            self.terrain_mode,
            target_name,
            self.state,
            target_distance_m,
            target_bearing_deg,
            terrain_height,
            wheel_slip,
            power_draw,
            path_curvature,
            target is None,
        )
        self.step_idx += 1
        self._sync_scene()

        if self._snapshot_path and not self._snapshot_taken:
            self.capture_snapshot(self._snapshot_path, exit_after_save=self._snapshot_exit)

        if self._snapshot_exit and self._snapshot_taken:
            self.finished = True
            self.userExit()
            return task.done

        if (self.max_steps is not None and self.step_idx >= self.max_steps) or self.finished:
            self.finished = True
            self.userExit()
            return task.done

        return task.cont

    def save_path_csv(self, path: str = "sim_rover_3d_path.csv") -> None:
        self.log.save_csv(path)


def run_rover_3d_demo(
    waypoints: Optional[Sequence[Waypoint3D]] = None,
    show_video: bool = True,
    save_csv: bool = True,
    delay_ms: int = 40,
    target_fps: float = 90.0,
    max_steps: Optional[int] = None,
    waypoint_tolerance_m: float = 0.45,
    control_mode: str = "pursuit",
    terrain_mode: str = "grassland",
    display_resolution: Tuple[int, int] = (2560, 1440),
    csv_path: str = "sim_rover_3d_path.csv",
    run_label: str = "demo",
    start_manual: bool = False,
    snapshot_path: Optional[str] = None,
    snapshot_exit: bool = False,
) -> MotionLog:
    if not show_video:
        log.info("Panda3D backend requires a GUI window; continuing with the interactive renderer.")

    normalized_resolution = normalize_resolution(display_resolution)
    app = Rover3DPandaApp(
        waypoints=waypoints,
        waypoint_tolerance_m=waypoint_tolerance_m,
        control_mode=control_mode,
        terrain_mode=terrain_mode,
        run_label=run_label,
        max_steps=max_steps,
        display_resolution=normalized_resolution,
        target_fps=clamp_target_fps(target_fps),
    )
    app.manual_mode = start_manual
    app._snapshot_path = snapshot_path
    app._snapshot_exit = snapshot_exit
    log.info("Starting Panda3D rover demo with %d waypoints", len(app.waypoints))
    if max_steps is None:
        log.info("Simulation end condition: mission complete or user quit")
    else:
        log.info("Simulation end condition: max_steps=%d, mission complete, or user quit", max_steps)
    log.info("Display size: %sx%s | target FPS: %.1f | legacy delay: %sms", normalized_resolution[0], normalized_resolution[1], app.target_fps, delay_ms)

    try:
        if snapshot_path and not snapshot_exit:
            app.taskMgr.doMethodLater(0.2, lambda task: (app.capture_snapshot(snapshot_path, exit_after_save=False), task.done)[1], "snapshot-once")
        app.run()
    finally:
        if save_csv:
            app.save_path_csv(csv_path)

    if app.log.rows:
        last = app.log.rows[-1]
        print("\nRover 3D Simulation Summary")
        print(f"  Steps            : {len(app.log.rows)}")
        print(f"  Final position   : x={last.x:+.2f} m  y={last.y:+.2f} m")
        print(f"  Final heading    : {last.heading_deg:+.1f} deg")
        print(f"  Final speed      : {last.speed:.2f} m/s")
        print(f"  Target complete  : {app.finished}")
        print()

    return app.log


def _build_waypoints_from_csv(path: str) -> List[Waypoint3D]:
    waypoints: List[Waypoint3D] = []
    with open(path, newline="") as file_handle:
        for row in csv.DictReader(file_handle):
            waypoints.append(Waypoint3D(name=row["name"], x=float(row["x"]), y=float(row["y"])))
    return waypoints


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

    parser = argparse.ArgumentParser(description="DIRT rover 3D simulation (Panda3D)")
    parser.add_argument("--no-gui", action="store_true", help="Retained for compatibility; Panda3D opens a window")
    parser.add_argument("--iters", type=int, default=0, help="Max simulation steps; 0 means no fixed step limit")
    parser.add_argument("--speed", type=int, default=40, help="Legacy frame delay ms (kept for compatibility)")
    parser.add_argument("--fps", type=float, default=90.0, help="Target render rate in FPS, between 60 and 120")
    parser.add_argument("--resolution", type=str, default="2560x1440", help="Display resolution as WIDTHxHEIGHT, e.g. 2560x1440")
    parser.add_argument("--waypoints", type=str, default=None, help="Optional CSV with columns name,x,y")
    parser.add_argument("--csv", type=str, default="sim_rover_3d_path.csv", help="CSV output path")
    parser.add_argument("--control-mode", choices=sorted(CONTROL_PROFILES.keys()), default="pursuit", help="Motion control profile")
    parser.add_argument("--terrain-mode", choices=sorted(TERRAIN_PROFILES), default="grassland", help="Terrain profile")
    parser.add_argument("--label", type=str, default="demo", help="Run label written into the CSV")
    parser.add_argument("--manual", action="store_true", help="Start in manual driving mode")
    parser.add_argument("--snapshot", type=str, default=None, help="Save a PNG snapshot to this path and exit after capture")
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
        max_steps=None if args.iters <= 0 else args.iters,
        control_mode=args.control_mode,
        terrain_mode=args.terrain_mode,
        display_resolution=(width, height),
        csv_path=args.csv,
        run_label=args.label,
        start_manual=args.manual,
        snapshot_path=args.snapshot,
        snapshot_exit=bool(args.snapshot),
    )