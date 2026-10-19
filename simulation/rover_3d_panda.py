"""
DIRT Bot - 3D Rover Simulation (Panda3D viewer)
===============================================

Draws the simulation from dirt_sim_core.py in 3D. All motion, control,
mission and logging logic lives in the core; this file only renders it and
handles keyboard/mouse input, so the CSV you get from the viewer and from a
headless run of the same scenario come from the exact same model.

    python rover3d_panda.py                          # interactive viewer
    python rover3d_panda.py --terrain-mode rocky --control-mode aggressive
    python rover3d_panda.py --terrain-mode testyard  # ramps from 5 to 40 deg
    python rover3d_panda.py --headless               # no window (same as dirt_sim_core.py)
    python rover3d_panda.py --snapshot shot.png      # render a frame and exit

Controls: click to add a waypoint, W/S throttle, A/D steer, M manual,
1 auto, 2 mission on/off, C camera, Space pause, R reset, X zero input, Q quit.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dirt_sim_core import (
    CONTROL_PROFILES,
    DEFAULT_CONTROL,
    DEFAULT_TERRAIN,
    TERRAIN_PROFILES,
    TerrainField,
    MotionLog,
    RoverParams,
    RoverSim,
    RoverState,
    Waypoint3D,
    add_common_args,
    clamp,
    testyard_lanes,
    TESTYARD_RUNUP_X,
    TESTYARD_HALF_WIDTH_M,
    params_from_args,
    print_summary,
    run_headless,
    waypoints_from_args,
)

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    CardMaker,
    ClockObject,
    DirectionalLight,
    Filename,
    Fog,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexArrayFormat,
    GeomVertexData,
    GeomVertexFormat,
    InternalName,
    LineSegs,
    LVector3f,
    Material,
    NodePath,
    Point3,
    TextNode,
    Texture,
    TransparencyAttrib,
    loadPrcFileData,
)

log = logging.getLogger("rover3d-panda")

# Re-exported so older scripts that did `from rover3d_panda import ...` still work.
__all__ = [
    "CONTROL_PROFILES", "TERRAIN_PROFILES", "TerrainField", "Waypoint3D", "RoverState",
    "MotionLog", "Rover3DPandaApp", "run_rover_3d_demo",
]

TRAIL_MAX_POINTS = 400
SIM_DT = 1.0 / 60.0          # physics step; rendering can run at any rate
MAX_SUBSTEPS = 6             # catch-up cap so a slow frame can't spiral

# Rover model geometry (model space: +X forward, +Y left, +Z up).
DRILL_REST_LIFT_M = 0.61      # raises the drill so the bit clears the ground at rest
DRILL_TRAVEL_M = 0.45         # how far the bit plunges below the surface
PROBE_TRAVEL_M = 0.95         # measurement probe drop to touch the soil


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _lerp(value: float, target: float, alpha: float) -> float:
    return value + (target - value) * clamp(alpha, 0.0, 1.0)


def normalize_resolution(value) -> Tuple[int, int]:
    """Normalize the requested window size and keep it in a sane range."""
    if value is None:
        return (2560, 1440)
    if isinstance(value, str):
        try:
            width_text, height_text = value.lower().split("x", 1)
            return max(640, int(width_text)), max(360, int(height_text))
        except ValueError:
            return (2560, 1440)
    width, height = value
    return max(640, int(width)), max(360, int(height))


def clamp_target_fps(value: float) -> float:
    fps = float(value)
    if fps <= 0:
        return 90.0
    return max(30.0, min(240.0, fps))


def lighting_profile(terrain_mode: str) -> dict:
    if terrain_mode == "testyard":
        return {
            "ambient": (0.36, 0.36, 0.38, 1.0),
            "sun_color": (1.00, 0.96, 0.90, 1.0),
            "fill_color": (0.30, 0.34, 0.44, 1.0),
            "background": (0.62, 0.74, 0.86, 1.0),
            "fog_color": (0.68, 0.78, 0.88),
            "fog_density": 0.008,
        }
    if terrain_mode == "farmland":
        return {
            "ambient": (0.30, 0.30, 0.33, 1.0),
            "sun_color": (1.00, 0.90, 0.74, 1.0),
            "fill_color": (0.30, 0.34, 0.46, 1.0),
            "background": (0.56, 0.70, 0.80, 1.0),
            "fog_color": (0.62, 0.74, 0.82),
            "fog_density": 0.010,
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
    return {
        "farmland": {"sun_glow": (1.00, 0.80, 0.46), "cloud_alpha": 0.30},
        "dunes": {"sun_glow": (1.00, 0.86, 0.60), "cloud_alpha": 0.22},
        "rocky": {"sun_glow": (0.96, 0.82, 0.58), "cloud_alpha": 0.18},
    }.get(terrain_mode, {"sun_glow": (1.00, 0.90, 0.72), "cloud_alpha": 0.24})


_FORMATS: Dict[Tuple[bool, bool], GeomVertexFormat] = {}


def _vertex_format(with_color: bool, with_uv: bool):
    key = (with_color, with_uv)
    if key not in _FORMATS:
        array = GeomVertexArrayFormat()
        array.addColumn(InternalName.getVertex(), 3, Geom.NTFloat32, Geom.CPoint)
        array.addColumn(InternalName.getNormal(), 3, Geom.NTFloat32, Geom.CNormal)
        if with_color:
            array.addColumn(InternalName.getColor(), 4, Geom.NTFloat32, Geom.CColor)
        if with_uv:
            array.addColumn(InternalName.getTexcoord(), 2, Geom.NTFloat32, Geom.CTexcoord)
        _FORMATS[key] = GeomVertexFormat.registerFormat(GeomVertexFormat(array))
    return _FORMATS[key]


def build_mesh(
    name: str,
    positions: np.ndarray,
    normals: np.ndarray,
    indices: np.ndarray,
    colors: Optional[np.ndarray] = None,
    uvs: Optional[np.ndarray] = None,
) -> Geom:
    """Build a Geom from numpy arrays in one bulk copy (no per-vertex Python loop)."""
    columns = [positions, normals]
    if colors is not None:
        columns.append(colors)
    if uvs is not None:
        columns.append(uvs)
    data = np.ascontiguousarray(np.hstack(columns), dtype=np.float32)

    vdata = GeomVertexData(name, _vertex_format(colors is not None, uvs is not None), Geom.UHStatic)
    vdata.uncleanSetNumRows(len(positions))
    memoryview(vdata.modifyArray(0)).cast("B")[:] = data.tobytes()

    tris = GeomTriangles(Geom.UHStatic)
    tris.setIndexType(Geom.NTUint32)
    index_array = tris.modifyVertices()
    index_array.uncleanSetNumRows(len(indices))
    memoryview(index_array).cast("B")[:] = np.ascontiguousarray(indices, dtype=np.uint32).tobytes()

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    return geom


def _unit_box_geom() -> Geom:
    faces = [
        ((0, 0, 1), [(-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)]),
        ((0, 0, -1), [(-.5, .5, -.5), (.5, .5, -.5), (.5, -.5, -.5), (-.5, -.5, -.5)]),
        ((0, 1, 0), [(-.5, .5, .5), (.5, .5, .5), (.5, .5, -.5), (-.5, .5, -.5)]),
        ((0, -1, 0), [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5)]),
        ((1, 0, 0), [(.5, -.5, .5), (.5, -.5, -.5), (.5, .5, -.5), (.5, .5, .5)]),
        ((-1, 0, 0), [(-.5, -.5, -.5), (-.5, -.5, .5), (-.5, .5, .5), (-.5, .5, -.5)]),
    ]
    positions, normals, indices = [], [], []
    for face_index, (normal, corners) in enumerate(faces):
        base = face_index * 4
        positions.extend(corners)
        normals.extend([normal] * 4)
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return build_mesh("box", np.array(positions), np.array(normals), np.array(indices))


def _rgba_texture(name: str, rgba: np.ndarray, repeat: bool = False) -> Texture:
    """Upload an (H, W, 4) float array in [0, 1] as a texture in one call."""
    height, width = rgba.shape[:2]
    pixels = (np.clip(rgba, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    texture = Texture(name)
    texture.setup2dTexture(width, height, Texture.TUnsignedByte, Texture.FRgba8)
    texture.setRamImageAs(np.ascontiguousarray(pixels[::-1]).tobytes(), "RGBA")
    if repeat:
        texture.setWrapU(Texture.WMRepeat)
        texture.setWrapV(Texture.WMRepeat)
    texture.setMagfilter(Texture.FTLinear)
    texture.setMinfilter(Texture.FTLinearMipmapLinear)
    return texture


# --------------------------------------------------------------------------
# The viewer
# --------------------------------------------------------------------------

class Rover3DPandaApp(ShowBase):
    def __init__(
        self,
        waypoints: Optional[Sequence[Waypoint3D]] = None,
        terrain_span_m: float = 48.0,
        waypoint_tolerance_m: float = 0.45,
        max_speed_mps: Optional[float] = None,
        steering_gain: Optional[float] = None,
        accel_mps2: Optional[float] = None,
        control_mode: str = DEFAULT_CONTROL,
        terrain_mode: str = DEFAULT_TERRAIN,
        run_label: str = "demo",
        max_steps: Optional[int] = None,
        display_resolution=(2560, 1440),
        target_fps: float = 90.0,
        params: Optional[RoverParams] = None,
        mission_mode: Optional[bool] = None,
        exit_on_complete: bool = False,
    ):
        self.display_resolution = normalize_resolution(display_resolution)
        self.target_fps = clamp_target_fps(target_fps)
        # Size the window up front; requesting it later leaves the first
        # frames (and any early snapshot) at Panda's 800x600 default.
        width, height = self.display_resolution
        loadPrcFileData("", f"win-size {width} {height}")
        loadPrcFileData("", f"window-title DIRT Rover Sim - {terrain_mode} / {control_mode}")
        loadPrcFileData("", "framebuffer-multisample 1\nmultisamples 4")
        loadPrcFileData("", "audio-library-name null")
        super().__init__()

        self.sim = RoverSim(
            waypoints=waypoints,
            terrain_mode=terrain_mode,
            control_mode=control_mode,
            terrain_span_m=terrain_span_m,
            waypoint_tolerance_m=waypoint_tolerance_m,
            max_speed_mps=max_speed_mps,
            steering_gain=steering_gain,
            accel_mps2=accel_mps2,
            params=params,
            mission_mode=mission_mode,
            run_label=run_label,
        )
        self.terrain = self.sim.terrain
        self.terrain_mode = self.sim.terrain_mode
        self.control_mode = self.sim.control_mode
        self.max_steps = max_steps
        self.exit_on_complete = exit_on_complete

        self.paused = False
        self.finished = False
        self._quit_requested = False
        self._accumulator = 0.0
        self._render_frame = 0
        self._camera_mode = "follow"
        self._camera_pos: Optional[LVector3f] = None
        self._camera_target: Optional[LVector3f] = None
        self._keys = {"w": False, "a": False, "s": False, "d": False}
        self._trail_points: List[Tuple[float, float, float]] = []
        self._trail_dirty = True
        self._trail_np: Optional[NodePath] = None
        self._route_np: Optional[NodePath] = None
        self.waypoint_nodes: List[NodePath] = []
        self._obstacle_nodes: Dict[str, NodePath] = {}
        self._detected_obstacles: set = set()
        self._nearest_obstacle_text = "none"
        self._tread_nodes: List[Tuple[NodePath, float, bool, str]] = []
        self._cloud_nodes: List[NodePath] = []
        self._snapshot_path: Optional[str] = None
        self._snapshot_exit = False
        self._snapshot_after_frames = 12
        self._snapshot_taken = False

        self._box_geom = _unit_box_geom()

        self.disableMouse()
        self.render.setAntialias(AntialiasAttrib.MMultisample)
        self.camLens.setFov(62)
        self.camLens.setNearFar(0.15, 2000.0)
        self._build_scene()
        self._build_hud()
        self._bind_controls()
        self._sync_scene(0.0)

        clock = ClockObject.getGlobalClock()
        clock.setMode(ClockObject.MLimited)
        clock.setFrameRate(self.target_fps)
        self.taskMgr.add(self._update_task, "rover3d-update")

    # Convenience pass-throughs so older code that poked at the app still works.
    @property
    def state(self) -> RoverState:
        return self.sim.state

    @property
    def waypoints(self) -> List[Waypoint3D]:
        return self.sim.waypoints

    @property
    def log(self) -> MotionLog:
        return self.sim.log

    # ------------------------------------------------------------- input

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
        self.accept("mouse1", self._place_waypoint)
        self.accept("f12", self.capture_snapshot)
        for key in self._keys:
            self.accept(key, self._set_key, [key, True])
            self.accept(f"{key}-up", self._set_key, [key, False])

    def _set_key(self, key: str, value: bool) -> None:
        self._keys[key] = value

    def _request_quit(self) -> None:
        self._quit_requested = True

    def toggle_manual_mode(self) -> None:
        self.sim.manual_mode = not self.sim.manual_mode

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def zero_manual_input(self) -> None:
        self.sim.manual_throttle = 0.0
        self.sim.manual_steer = 0.0

    def _set_auto_mode(self) -> None:
        self.sim.manual_mode = False
        self.zero_manual_input()

    def _toggle_mission_mode(self) -> None:
        self.sim.set_mission_mode(not self.sim.mission_mode)

    def _cycle_camera(self) -> None:
        modes = ("follow", "overhead", "cockpit", "side")
        self._camera_mode = modes[(modes.index(self._camera_mode) + 1) % len(modes)]

    def reset_pose(self) -> None:
        self.sim.reset()
        self.finished = False
        self._trail_points.clear()
        self._trail_dirty = True
        self._detected_obstacles.clear()
        self._refresh_waypoints()

    def _terrain_hit(self, start: Point3, direction: LVector3f) -> Optional[Tuple[float, float]]:
        """March a mouse ray until it dips below the terrain, then bisect."""
        length = direction.length()
        if length < 1e-9:
            return None
        direction = direction / length
        step, t_prev, t = 0.5, 0.0, 0.0
        above_prev = True
        for _ in range(2000):
            t += step
            p = start + direction * t
            above = p.z > float(self.terrain.height(p.x, p.y))
            if above_prev and not above:
                lo, hi = t_prev, t
                for _ in range(24):
                    mid = 0.5 * (lo + hi)
                    q = start + direction * mid
                    if q.z > float(self.terrain.height(q.x, q.y)):
                        lo = mid
                    else:
                        hi = mid
                q = start + direction * hi
                return float(q.x), float(q.y)
            above_prev, t_prev = above, t
            if t > 400.0:
                break
        return None

    def _place_waypoint(self) -> None:
        if not self.mouseWatcherNode.hasMouse():
            return
        mouse = self.mouseWatcherNode.getMouse()
        near, far = Point3(), Point3()
        if not self.camLens.extrude(mouse, near, far):
            return
        world_near = self.render.getRelativePoint(self.camera, near)
        world_far = self.render.getRelativePoint(self.camera, far)
        hit = self._terrain_hit(world_near, world_far - world_near)
        if hit is None:
            return
        x, y = hit
        half = self.terrain.span_m / 2.0 - self.sim.params.footprint_radius_m
        if not (-half <= x <= half and -half <= y <= half):
            return
        self.sim.add_waypoint(x, y)
        self.finished = False
        self._rebuild_waypoint_visuals()

    # ------------------------------------------------------------- building

    def _build_box(self, name: str, color) -> NodePath:
        node = GeomNode(name)
        node.addGeom(self._box_geom)  # shared geometry, per-node transform/colour
        np_node = NodePath(node)
        np_node.setColor(*color)
        return np_node

    def _apply_material(self, node: NodePath, diffuse, ambient, specular, shininess: float) -> None:
        material = Material()
        material.setDiffuse(diffuse)
        material.setAmbient(ambient)
        material.setSpecular(specular)
        material.setShininess(shininess)
        node.setMaterial(material, 1)

    def _part(self, parent: NodePath, name: str, scale, pos, diffuse, ambient, specular, shininess, hpr=None) -> NodePath:
        part = self._build_box(name, diffuse)
        part.reparentTo(parent)
        part.setScale(*scale)
        part.setPos(*pos)
        if hpr is not None:
            part.setHpr(*hpr)
        self._apply_material(part, diffuse, ambient, specular, shininess)
        return part

    def _build_scene(self) -> None:
        self._setup_lights()
        self._sky_np = self._make_sky_dome()
        self.terrain_np = self._make_terrain()
        self._build_ground_cover()
        self._build_rover()
        self._spawn_field_obstacles()
        self._build_sky_sprites()
        self._rebuild_waypoint_visuals()

    def _setup_lights(self) -> None:
        self.render.setShaderAuto()
        profile = lighting_profile(self.terrain_mode)

        fog = Fog("atmosphere")
        fog.setColor(*profile["fog_color"])
        fog.setExpDensity(profile["fog_density"])
        self.render.setFog(fog)

        ambient = AmbientLight("ambient")
        ambient.setColor(profile["ambient"])
        self.render.setLight(self.render.attachNewNode(ambient))

        key_light = DirectionalLight("sun")
        key_light.setColor(profile["sun_color"])
        key_np = self.render.attachNewNode(key_light)
        key_np.setHpr(-35, -52, 0)
        self.render.setLight(key_np)

        fill_light = DirectionalLight("fill")
        fill_light.setColor(profile["fill_color"])
        fill_np = self.render.attachNewNode(fill_light)
        fill_np.setHpr(145, -26, 0)
        self.render.setLight(fill_np)

        self.setBackgroundColor(*profile["background"])

    def _terrain_colors(self, gx: np.ndarray, gy: np.ndarray, z: np.ndarray) -> np.ndarray:
        height_t = np.clip((z + 0.8) / 1.7, 0.0, 1.0)[..., None]
        mode = self.terrain_mode
        if mode == "testyard":
            # Concrete pad with a 2 m grid, ramps coloured green -> red by angle.
            grid = (np.abs(((gx + 1.0) % 2.0) - 1.0) < 0.04) | (np.abs(((gy + 1.0) % 2.0) - 1.0) < 0.04)
            rgb = np.where(grid[..., None], np.array([0.46, 0.46, 0.45]), np.array([0.66, 0.65, 0.62]))
            angle = self.terrain.ramp_angle(gx, gy)
            on_ramp = z > 0.02
            t = np.clip((angle - 5.0) / 35.0, 0.0, 1.0)[..., None]
            ramp = np.where(
                t < 0.5,
                np.array([0.20, 0.62, 0.30]) * (1 - 2 * t) + np.array([0.92, 0.78, 0.18]) * (2 * t),
                np.array([0.92, 0.78, 0.18]) * (2 - 2 * t) + np.array([0.86, 0.22, 0.16]) * (2 * t - 1),
            )
            # Grip strips every 0.5 m across the slope faces.
            strip = (np.abs(((gx + 0.25) % 0.5) - 0.25) < 0.05)[..., None]
            ramp = np.where(strip, ramp * 0.72, ramp)
            return np.where(on_ramp[..., None], ramp, rgb)
        if mode == "farmland":
            # Bands line up with the furrow ridges in TerrainField.height().
            ridge = 0.5 + 0.5 * np.sin(gy * TerrainField.FURROW_K)
            lane = 0.5 + 0.5 * np.sin((gx - 0.28 * gy) * 0.35)
            dark = np.array([0.30, 0.21, 0.12])
            mid = np.array([0.46, 0.34, 0.19])
            light = np.array([0.58, 0.46, 0.27])
            rgb = np.where(ridge[..., None] > 0.62, light, np.where(ridge[..., None] > 0.30, mid, dark))
            # Large-scale moisture variation so the field isn't a flat stripe pattern.
            rgb = rgb * (0.88 + 0.12 * lane[..., None]) + 0.10 * height_t
            return rgb
        if mode == "dunes":
            base, top = np.array([0.66, 0.53, 0.32]), np.array([0.96, 0.84, 0.58])
        elif mode == "rocky":
            base, top = np.array([0.30, 0.28, 0.26]), np.array([0.64, 0.60, 0.54])
        elif mode == "grassland":
            base, top = np.array([0.24, 0.34, 0.16]), np.array([0.46, 0.58, 0.26])
        else:  # rolling
            base, top = np.array([0.30, 0.36, 0.20]), np.array([0.66, 0.62, 0.40])
        return base * (1.0 - height_t) + top * height_t

    def _make_detail_texture(self) -> Texture:
        """Neutral grit texture; hue comes from vertex colours, so it suits every terrain."""
        size = 256
        rng = np.random.default_rng(17)
        fine = rng.random((size, size))
        coarse = np.repeat(np.repeat(rng.random((32, 32)), 8, axis=0), 8, axis=1)
        lum = 0.84 + 0.10 * fine + 0.08 * coarse
        lum = np.where(fine > 0.97, lum * 0.72, lum)  # dark specks (stones, clods)
        lum = np.where(fine < 0.02, lum * 1.10, lum)  # light specks
        rgba = np.dstack([lum, lum, lum, np.ones_like(lum)])
        return _rgba_texture("terrain-detail", rgba, repeat=True)

    def _make_terrain(self, spacing_m: float = 0.25) -> NodePath:
        """Terrain mesh at 25 cm spacing (fine enough to resolve the furrows)."""
        half = self.terrain.span_m / 2.0
        n = int(round(self.terrain.span_m / spacing_m)) + 1
        xs = np.linspace(-half, half, n)
        gx, gy, z, sx, sy = self.terrain.height_and_gradient_grid(xs, xs)

        normals = np.dstack([-sx, -sy, np.ones_like(z)])
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        rgb = self._terrain_colors(gx, gy, z)
        colors = np.dstack([rgb, np.ones_like(z)])
        uvs = np.dstack([gx / 2.4, gy / 2.4])

        grid = np.arange(n * n).reshape(n, n)
        a, b = grid[:-1, :-1].ravel(), grid[:-1, 1:].ravel()
        c, d = grid[1:, 1:].ravel(), grid[1:, :-1].ravel()
        indices = np.stack([a, b, c, a, c, d], axis=1).ravel()

        geom = build_mesh(
            "terrain",
            np.dstack([gx, gy, z]).reshape(-1, 3),
            normals.reshape(-1, 3),
            indices,
            colors=colors.reshape(-1, 4),
            uvs=uvs.reshape(-1, 2),
        )
        node = GeomNode("terrain")
        node.addGeom(geom)
        terrain_np = self.render.attachNewNode(node)
        terrain_np.setTexture(self._make_detail_texture(), 1)
        return terrain_np

    def _build_ground_cover(self) -> None:
        """Crops / grass / pebbles / fence, all flattened into a few draw calls."""
        root = self.render.attachNewNode("ground-cover")
        rng = random.Random(11)
        half = self.terrain.span_m / 2.0 - 1.0

        def tuft(x: float, y: float, height: float, color, lean: float) -> None:
            node = self._build_box("tuft", color)
            node.reparentTo(root)
            node.setScale(0.05, 0.05, height)
            node.setPos(x, y, float(self.terrain.height(x, y)) + height * 0.45)
            node.setHpr(rng.uniform(0, 90), 0.0, lean)

        mode = self.terrain_mode
        if mode == "farmland":
            # One crop row on every second furrow ridge.
            k = TerrainField.FURROW_K
            first = math.ceil((-half * k - math.pi / 2) / (2 * math.pi))
            last = math.floor((half * k - math.pi / 2) / (2 * math.pi))
            for row in range(first, last + 1, 2):
                y = (math.pi / 2 + 2 * math.pi * row) / k
                x = -half
                while x < half:
                    green = (0.20 + rng.uniform(0, 0.06), 0.44 + rng.uniform(0, 0.10), 0.12, 1.0)
                    tuft(x + rng.uniform(-0.15, 0.15), y, rng.uniform(0.16, 0.28), green, rng.uniform(-12, 12))
                    x += rng.uniform(0.9, 1.5)
        elif mode in ("grassland", "rolling"):
            for _ in range(700):
                green = (0.20 + rng.uniform(0, 0.08), 0.42 + rng.uniform(0, 0.12), 0.12, 1.0)
                tuft(rng.uniform(-half, half), rng.uniform(-half, half), rng.uniform(0.14, 0.30), green, rng.uniform(-18, 18))
        elif mode == "rocky":
            for _ in range(450):
                x, y = rng.uniform(-half, half), rng.uniform(-half, half)
                s = rng.uniform(0.06, 0.18)
                grey = rng.uniform(0.34, 0.55)
                pebble = self._build_box("pebble", (grey, grey * 0.97, grey * 0.92, 1.0))
                pebble.reparentTo(root)
                pebble.setScale(s * 1.3, s, s * 0.6)
                pebble.setPos(x, y, float(self.terrain.height(x, y)) + s * 0.15)
                pebble.setHpr(rng.uniform(0, 360), rng.uniform(-15, 15), rng.uniform(-15, 15))

        if mode == "testyard":
            orange = ((1.0, 0.45, 0.08, 1.0), (0.30, 0.12, 0.02, 1.0), (0.6, 0.5, 0.4, 1.0), 20.0)
            for _angle, yc in testyard_lanes():
                for x in (-7.0, 7.0):
                    for side in (-1.0, 1.0):
                        y = yc + side * (TESTYARD_HALF_WIDTH_M + 0.4)
                        self._part(root, "cone-base", (0.30, 0.30, 0.04), (x, y, 0.02), *orange)
                        self._part(root, "cone", (0.16, 0.16, 0.40), (x, y, 0.22), *orange)
                        self._part(root, "cone-band", (0.17, 0.17, 0.06), (x, y, 0.28),
                                   (0.95, 0.95, 0.95, 1.0), (0.3, 0.3, 0.3, 1.0), (0.5, 0.5, 0.5, 1.0), 20.0)

        if mode in ("farmland", "grassland", "rolling"):
            edge = self.terrain.span_m / 2.0 - 1.6
            for n in range(16):
                for x, y in ((-22.5 + n * 3.0, -edge), (-edge, -22.0 + n * 3.0)):
                    post = self._build_box("fence-post", (0.62, 0.52, 0.34, 1.0))
                    post.reparentTo(root)
                    post.setScale(0.12, 0.12, 1.0)
                    post.setPos(x, y, float(self.terrain.height(x, y)) + 0.45)
            # Rails between posts, following the ground.
            for axis in ("x", "y"):
                rail = LineSegs(f"fence-rail-{axis}")
                rail.setThickness(3.0)
                rail.setColor(0.55, 0.45, 0.30, 1.0)
                for height in (0.55, 0.85):
                    for i, s in enumerate(np.linspace(-22.5, 22.5, 61)):
                        x, y = (s, -edge) if axis == "x" else (-edge, s)
                        p = (x, y, float(self.terrain.height(x, y)) + height)
                        rail.moveTo(*p) if i == 0 else rail.drawTo(*p)
                root.attachNewNode(rail.create())

        root.flattenStrong()
        if mode == "testyard":
            self._build_testyard_signs()

    def _build_testyard_signs(self) -> None:
        for angle, yc in testyard_lanes():
            for x in (-TESTYARD_RUNUP_X - 2.2, TESTYARD_RUNUP_X + 2.2):
                post = self._build_box("sign-post", (0.30, 0.30, 0.32, 1.0))
                post.reparentTo(self.render)
                post.setScale(0.08, 0.08, 1.4)
                post.setPos(x, yc + TESTYARD_HALF_WIDTH_M + 0.6, 0.7)
                text = TextNode(f"sign-{angle:g}")
                text.setText(f"{angle:g} deg")
                text.setAlign(TextNode.ACenter)
                text.setTextColor(1.0, 1.0, 1.0, 1.0)
                text.setCardColor(0.10, 0.12, 0.16, 0.92)
                text.setCardAsMargin(0.25, 0.25, 0.15, 0.15)
                text.setCardDecal(True)
                label = self.render.attachNewNode(text)
                label.setScale(0.55)
                label.setPos(x, yc + TESTYARD_HALF_WIDTH_M + 0.6, 1.55)
                label.setBillboardPointEye()
                label.setLightOff(1)
                label.setFogOff(1)

    def _make_sky_dome(self, radius: float = 140.0, slices: int = 48, stacks: int = 24) -> NodePath:
        phi = np.linspace(0.0, math.pi / 2.0, stacks + 1)
        theta = np.linspace(0.0, 2.0 * math.pi, slices + 1)
        th, ph = np.meshgrid(theta, phi)
        x, y, z = np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)
        mix = (ph / (math.pi / 2.0))[..., None]
        horizon = np.array([0.92, 0.82, 0.66])
        mid = np.array([0.55, 0.74, 0.95])
        zenith = np.array([0.22, 0.42, 0.78])
        # mix 0 = zenith, 1 = horizon.
        t1 = np.clip(mix / 0.6, 0.0, 1.0)
        t2 = np.clip((mix - 0.6) / 0.4, 0.0, 1.0)
        rgb = np.where(mix < 0.6, zenith * (1 - t1) + mid * t1, mid * (1 - t2) + horizon * t2)

        row = slices + 1
        grid = np.arange((stacks + 1) * row).reshape(stacks + 1, row)
        a, b = grid[:-1, :-1].ravel(), grid[:-1, 1:].ravel()
        c, d = grid[1:, 1:].ravel(), grid[1:, :-1].ravel()
        indices = np.stack([a, b, c, a, c, d], axis=1).ravel()
        positions = np.dstack([x, y, z]).reshape(-1, 3) * radius
        geom = build_mesh(
            "sky", positions, -positions / radius, indices,
            colors=np.dstack([rgb, np.ones(x.shape)]).reshape(-1, 4),
        )
        node = GeomNode("sky")
        node.addGeom(geom)
        sky_np = self.render.attachNewNode(node)
        sky_np.setTwoSided(True)
        sky_np.setLightOff(1)
        sky_np.setFogOff(1)
        sky_np.setShaderOff(1)
        sky_np.setDepthWrite(False)
        sky_np.setBin("background", 0)
        return sky_np

    def _make_soft_disc_texture(self, name: str, core, edge) -> Texture:
        size = 128
        idx = (np.arange(size) - 63.5) / 63.5
        gx, gy = np.meshgrid(idx, idx)
        alpha = np.clip(1.0 - np.sqrt(gx ** 2 + gy ** 2), 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        rgba = np.dstack([
            edge[0] * (1 - alpha) + core[0] * alpha,
            edge[1] * (1 - alpha) + core[1] * alpha,
            edge[2] * (1 - alpha) + core[2] * alpha,
            edge[3] * (1 - alpha) + alpha,
        ])
        return _rgba_texture(name, rgba)

    def _make_billboard_sprite(self, name: str, texture: Texture, scale: float, color) -> NodePath:
        card = CardMaker(name)
        card.setFrame(-1.0, 1.0, -1.0, 1.0)
        node = self.render.attachNewNode(card.generate())
        node.setBillboardPointEye()
        node.setScale(scale)
        node.setTexture(texture)
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setColor(*color)
        node.setLightOff(1)
        node.setFogOff(1)
        node.setShaderOff(1)
        node.setDepthWrite(False)
        node.setBin("background", 5)
        return node

    def _build_sky_sprites(self) -> None:
        weather = weather_profile(self.terrain_mode)
        glow = weather["sun_glow"]
        sun_texture = self._make_soft_disc_texture("sun", glow, (*glow, 0.0))
        self._sun_np = self._make_billboard_sprite("sun-disc", sun_texture, 16.0, (1.0, 0.96, 0.82, 0.9))
        cloud_texture = self._make_soft_disc_texture("cloud", (0.98, 0.98, 1.0), (0.96, 0.98, 1.0, 0.0))
        for index in range(3):
            cloud = self._make_billboard_sprite(
                f"cloud-{index}", cloud_texture, 18.0 + index * 3.5, (1.0, 1.0, 1.0, weather["cloud_alpha"]),
            )
            self._cloud_nodes.append(cloud)

    # ------------------------------------------------------------- rover model

    def _build_rover(self) -> None:
        """Rover model. Pose node gets the sim pose; the model inside it is
        modelled +X forward, so it is turned 90 deg to Panda's +Y-forward
        convention and lifted so the tread bottoms sit on the ground."""
        self.rover_np = self.render.attachNewNode("rover")
        model = self.rover_np.attachNewNode("rover-model")
        model.setH(90.0)
        self._rover_model = model
        P = self._part

        dark = ((0.06, 0.08, 0.10, 1.0), (0.03, 0.04, 0.05, 1.0), (0.10, 0.12, 0.14, 1.0))
        P(model, "solar-deck", (1.88, 1.02, 0.04), (-0.02, 0.0, 0.92), *dark, 6.0)
        for idx, x_pos in enumerate([-0.72, -0.24, 0.24, 0.72]):
            P(model, f"solar-tile-{idx}", (0.22, 0.86, 0.010), (x_pos, 0.0, 0.965),
              (0.05, 0.08, 0.16, 1.0), (0.02, 0.03, 0.06, 1.0), (0.40, 0.46, 0.60, 1.0), 60.0)
        P(model, "top-spine", (1.32, 0.22, 0.06), (-0.02, 0.0, 0.78),
          (0.70, 0.72, 0.76, 1.0), (0.22, 0.24, 0.26, 1.0), (0.88, 0.90, 0.92, 1.0), 26.0)
        frame = ((0.18, 0.16, 0.12, 1.0), (0.08, 0.07, 0.05, 1.0), (0.24, 0.20, 0.16, 1.0))
        P(model, "deck-frame-front", (1.94, 0.06, 0.04), (-0.02, 0.52, 0.92), *frame, 10.0)
        P(model, "deck-frame-back", (1.94, 0.06, 0.04), (-0.02, -0.52, 0.92), *frame, 10.0)
        body = ((0.70, 0.71, 0.72, 1.0), (0.30, 0.31, 0.32, 1.0), (0.55, 0.56, 0.58, 1.0))
        P(model, "front-nose", (0.44, 0.28, 0.18), (1.02, 0.0, 0.56), *body, 30.0)
        P(model, "front-beam", (0.48, 0.16, 0.10), (0.86, 0.0, 0.40), *body, 24.0)
        P(model, "center-body", (1.30, 0.76, 0.20), (0.00, 0.0, 0.42), *body, 28.0)
        P(model, "belly-pan", (1.08, 0.60, 0.06), (0.00, 0.0, 0.22),
          (0.08, 0.09, 0.11, 1.0), (0.04, 0.05, 0.05, 1.0), (0.16, 0.18, 0.20, 1.0), 12.0)
        P(model, "left-sponson", (0.60, 0.14, 0.10), (-0.02, 0.54, 0.28), *body, 18.0)
        P(model, "right-sponson", (0.60, 0.14, 0.10), (-0.02, -0.54, 0.28), *body, 18.0)
        P(model, "rear-box", (0.66, 0.52, 0.22), (-0.92, 0.0, 0.60),
          (0.78, 0.76, 0.72, 1.0), (0.32, 0.31, 0.29, 1.0), (0.60, 0.60, 0.58, 1.0), 40.0)
        P(model, "spectrometer-box", (0.30, 0.20, 0.14), (-0.88, 0.0, 0.72),
          (0.86, 0.66, 0.22, 1.0), (0.08, 0.10, 0.12, 1.0), (0.50, 0.56, 0.60, 1.0), 44.0)
        P(model, "spectrometer-lid", (0.30, 0.20, 0.05), (-0.88, 0.0, 0.88),
          (0.32, 0.38, 0.44, 1.0), (0.10, 0.12, 0.14, 1.0), (0.56, 0.60, 0.66, 1.0), 56.0)
        P(model, "stop-bracket", (0.09, 0.24, 0.16), (-1.18, 0.0, 0.46),
          (0.22, 0.22, 0.24, 1.0), (0.08, 0.08, 0.08, 1.0), (0.34, 0.36, 0.38, 1.0), 20.0)
        mast = ((0.88, 0.88, 0.90, 1.0), (0.32, 0.32, 0.34, 1.0), (0.66, 0.68, 0.70, 1.0))
        P(model, "mast-base", (0.07, 0.07, 0.20), (-0.06, 0.0, 1.00), *mast, 32.0)
        P(model, "mast", (0.05, 0.05, 0.72), (-0.06, 0.0, 1.34), *mast, 42.0)
        teal = ((0.16, 0.94, 0.82, 1.0), (0.04, 0.18, 0.16, 1.0), (0.66, 1.0, 0.94, 1.0))
        P(model, "mast-head", (0.12, 0.12, 0.12), (-0.06, 0.0, 1.84), *teal, 70.0)
        P(model, "mast-sensor", (0.12, 0.12, 0.05), (-0.06, 0.0, 1.96), *teal, 90.0)
        P(model, "camera-pod", (0.18, 0.14, 0.10), (0.28, 0.0, 0.98),
          (0.14, 0.16, 0.18, 1.0), (0.06, 0.07, 0.08, 1.0), (0.46, 0.50, 0.55, 1.0), 48.0)
        P(model, "camera-lens", (0.06, 0.06, 0.06), (0.42, 0.0, 0.98),
          (0.10, 0.90, 0.86, 1.0), (0.05, 0.22, 0.20, 1.0), (0.85, 1.0, 1.0, 1.0), 96.0)
        lamp = ((1.0, 0.90, 0.58, 1.0), (0.16, 0.12, 0.08, 1.0), (1.0, 0.98, 0.84, 1.0))
        for side, y in (("left", 0.22), ("right", -0.22)):
            light = P(model, f"front-light-{side}", (0.08, 0.05, 0.05), (1.26, y, 0.48), *lamp, 120.0)
            light.setLightOff(1)  # emissive look
        P(model, "antenna", (0.04, 0.04, 0.70), (-0.04, -0.34, 1.86), *mast, 42.0)
        P(model, "antenna-tip", (0.10, 0.10, 0.10), (-0.04, -0.34, 2.34), *teal, 72.0)

        # Tank treads. Each side: frame, top and bottom runs, end caps, moving
        # cleats, and road wheels inside the loop.
        tracks_root = model.attachNewNode("tracks")
        frame_mat = ((0.42, 0.44, 0.47, 1.0), (0.16, 0.17, 0.18, 1.0), (0.50, 0.52, 0.55, 1.0))
        belt = ((0.06, 0.07, 0.08, 1.0), (0.02, 0.02, 0.03, 1.0), (0.14, 0.16, 0.18, 1.0))
        for side_sign, side_name, y_pos in ((1.0, "left", 0.94), (-1.0, "right", -0.94)):
            P(model, f"side-mount-{side_name}", (0.12, 0.20, 0.12), (0.64, y_pos * 0.66, 0.28), *frame_mat, 24.0)
            P(model, f"side-strut-{side_name}", (0.62, 0.08, 0.08), (0.12, y_pos * 0.52, 0.20), *frame_mat, 20.0,
              hpr=(0.0, 0.0, side_sign * 4.0))
            sus = tracks_root.attachNewNode(f"suspension-{side_name}")
            sus.setPos(0.0, y_pos, 0.0)
            P(sus, "suspension-connector", (0.18, 0.08, 0.08), (0.66, 0.0, 0.12), *frame_mat, 24.0)
            P(sus, "rocker", (1.72, 0.07, 0.08), (-0.10, 0.0, 0.10), *frame_mat, 24.0)
            for leg, x_pos, h, z in (("front", 0.76, 0.36, -0.10), ("middle", 0.0, 0.34, -0.14), ("rear", -0.78, 0.36, -0.12)):
                P(sus, f"rocker-{leg}-leg", (0.10, 0.08, h), (x_pos, 0.0, z), *frame_mat, 20.0)
            P(sus, "track-top", (1.92, 0.32, 0.12), (0.0, 0.0, 0.40), *belt, 8.0)
            P(sus, "track-bottom", (1.92, 0.32, 0.12), (0.0, 0.0, -0.48), *belt, 8.0)
            for end, x_pos in (("front", 0.96), ("rear", -0.96)):
                P(sus, f"track-end-{end}", (0.12, 0.32, 0.44), (x_pos, 0.0, -0.04), *belt, 8.0)
            cleat = ((0.14, 0.15, 0.16, 1.0), (0.04, 0.04, 0.05, 1.0), (0.24, 0.26, 0.28, 1.0))
            for i, x_pos in enumerate(np.arange(-0.9, 0.9, 0.2)):  # even spacing around the 1.8 m loop
                top = P(sus, f"cleat-top-{i}", (0.07, 0.36, 0.05), (x_pos, 0.0, 0.47), *cleat, 10.0)
                bottom = P(sus, f"cleat-bottom-{i}", (0.07, 0.36, 0.05), (x_pos, 0.0, -0.555), *cleat, 10.0)
                self._tread_nodes.append((top, float(x_pos), True, side_name))
                self._tread_nodes.append((bottom, float(x_pos), False, side_name))
            for x_pos, label in ((0.82, "front"), (0.00, "middle"), (-0.82, "rear")):
                P(sus, f"wheel-{label}", (0.50, 0.30, 0.50), (x_pos, 0.0, -0.04),
                  (0.12, 0.12, 0.14, 1.0), (0.04, 0.04, 0.05, 1.0), (0.32, 0.32, 0.34, 1.0), 16.0)
                P(sus, f"wheel-hub-{label}", (0.20, 0.34, 0.20), (x_pos, 0.0, 0.08),
                  (0.48, 0.50, 0.54, 1.0), (0.18, 0.18, 0.20, 1.0), (0.80, 0.82, 0.85, 1.0), 80.0)
                P(sus, f"wheel-cap-{label}", (0.08, 0.36, 0.08), (x_pos, 0.0, 0.11),
                  (0.98, 0.88, 0.54, 1.0), (0.30, 0.24, 0.10, 1.0), (1.0, 0.96, 0.78, 1.0), 120.0)

        # Lift the model so the lowest tread cleat touches z = 0.
        bounds = tracks_root.getTightBounds(model)
        self._ride_height = -float(bounds[0].z) if bounds else 0.58
        model.setZ(self._ride_height)

        # Drill stage (front). Rest position keeps the bit just above the soil.
        self._drill_stage_np = model.attachNewNode("drill-stage")
        steel = ((0.32, 0.34, 0.38, 1.0), (0.12, 0.13, 0.14, 1.0), (0.56, 0.60, 0.64, 1.0))
        gold = ((0.95, 0.80, 0.22, 1.0), (0.22, 0.16, 0.05, 1.0), (1.0, 0.90, 0.46, 1.0))
        D = self._drill_stage_np
        P(D, "drill-base", (0.16, 0.14, 0.22), (1.30, 0.0, 0.02), *steel, 34.0)
        P(D, "drill-forearm", (0.09, 0.07, 0.54), (1.38, 0.0, -0.26), *steel, 40.0)
        P(D, "drill-joint", (0.10, 0.10, 0.10), (1.38, 0.0, -0.56),
          (0.94, 0.80, 0.30, 1.0), (0.22, 0.18, 0.07, 1.0), (1.0, 0.92, 0.48, 1.0), 86.0)
        P(D, "drill-guard", (0.16, 0.09, 0.12), (1.38, 0.0, -0.68),
          (0.18, 0.20, 0.24, 1.0), (0.06, 0.07, 0.08, 1.0), (0.34, 0.38, 0.42, 1.0), 20.0)
        bit = D.attachNewNode("drill-bit")
        bit.setPos(1.38, 0.0, 0.0)
        P(bit, "drill", (0.07, 0.07, 0.36), (0.0, 0.0, -0.90), *gold, 92.0)
        # Flutes: offset blades so the spin actually reads on screen.
        for i in range(3):
            P(bit, f"drill-flute-{i}", (0.12, 0.025, 0.06), (0.0, 0.0, -0.78 - i * 0.11), *gold, 92.0,
              hpr=(i * 60.0, 0.0, 0.0))
        P(bit, "drill-tip", (0.04, 0.04, 0.10), (0.0, 0.0, -1.10),
          (1.0, 0.92, 0.58, 1.0), (0.30, 0.24, 0.08, 1.0), (1.0, 0.98, 0.70, 1.0), 120.0)
        self._drill_bit_np = bit
        self._drill_rest_z = DRILL_REST_LIFT_M

        # Measurement probe (rear), DRILL_TO_PROBE_M behind the drill bit.
        self._measure_stage_np = model.attachNewNode("measure-stage")
        M = self._measure_stage_np
        P(M, "sensor-box", (0.26, 0.18, 0.12), (-0.82, 0.0, 0.40),
          (0.24, 0.28, 0.32, 1.0), (0.08, 0.10, 0.12, 1.0), (0.44, 0.48, 0.52, 1.0), 36.0)
        P(M, "sensor-lid", (0.26, 0.18, 0.05), (-0.82, 0.0, 0.48),
          (0.34, 0.38, 0.42, 1.0), (0.10, 0.12, 0.14, 1.0), (0.56, 0.60, 0.66, 1.0), 54.0)
        P(M, "sensor-window", (0.10, 0.10, 0.02), (-0.82, 0.0, 0.335), *teal, 90.0)

        self._shadow_np = self._make_rover_shadow()
        # Child of the pose node so it tilts with the rover on slopes.
        self._shadow_np.reparentTo(self.rover_np)
        self._shadow_np.setPosHpr(0.0, 0.0, 0.02, 0.0, -90.0, 0.0)

    def _make_rover_shadow(self) -> NodePath:
        card = CardMaker("rover-shadow")
        card.setFrame(-1.15, 1.15, -1.2, 1.2)
        shadow = self.render.attachNewNode(card.generate())
        shadow.setTransparency(TransparencyAttrib.MAlpha)
        shadow.setColor(0.0, 0.0, 0.0, 0.30)
        shadow.setLightOff(1)
        shadow.setDepthWrite(False)
        shadow.setDepthOffset(3)
        shadow.setBin("transparent", 10)
        return shadow

    # ------------------------------------------------------------- obstacles

    def _spawn_field_obstacles(self) -> None:
        for obs in self.sim.obstacles:
            node = self._build_tree(obs.name, obs.scale) if obs.kind == "tree" else self._build_rock(obs.name, obs.scale)
            node.setPos(obs.x, obs.y, float(self.terrain.height(obs.x, obs.y)))
            node.flattenStrong()
            self._obstacle_nodes[obs.name] = node

    def _build_tree(self, name: str, scale: float) -> NodePath:
        root = self.render.attachNewNode(name)
        bark = ((0.40, 0.26, 0.12, 1.0), (0.16, 0.10, 0.05, 1.0), (0.52, 0.34, 0.18, 1.0))
        self._part(root, "trunk", (0.14 * scale, 0.14 * scale, 0.72 * scale), (0.0, 0.0, 0.32 * scale), *bark, 10.0,
                   hpr=(-3.0, 0.0, 4.0))
        for side, x, z, rotation in (("left", -0.22, 0.78, -18.0), ("right", 0.20, 0.82, 20.0)):
            self._part(root, f"branch-{side}", (0.40 * scale, 0.07 * scale, 0.07 * scale), (x * scale, 0.0, z * scale),
                       (0.34, 0.20, 0.08, 1.0), (0.12, 0.07, 0.03, 1.0), (0.44, 0.26, 0.10, 1.0), 8.0,
                       hpr=(0.0, 0.0, rotation))
        foliage = [
            ("lower", 0.42, 0.34, 0.25, -0.02, 0.0, 0.90, (0.12, 0.34, 0.10, 1.0)),
            ("left", 0.30, 0.28, 0.25, -0.28, 0.02, 1.08, (0.16, 0.46, 0.14, 1.0)),
            ("right", 0.31, 0.27, 0.24, 0.27, -0.04, 1.10, (0.20, 0.52, 0.16, 1.0)),
            ("top", 0.25, 0.24, 0.24, 0.02, 0.04, 1.30, (0.26, 0.58, 0.18, 1.0)),
        ]
        for part, sx, sy, sz, x, y, z, diffuse in foliage:
            ambient = tuple(c * 0.42 for c in diffuse[:3]) + (1.0,)
            specular = tuple(min(1.0, c * 1.3) for c in diffuse[:3]) + (1.0,)
            self._part(root, f"canopy-{part}", (sx * scale, sy * scale, sz * scale), (x * scale, y * scale, z * scale),
                       diffuse, ambient, specular, 20.0, hpr=((x + y) * 18.0, y * 12.0, x * 20.0))
        return root

    def _build_rock(self, name: str, scale: float) -> NodePath:
        root = self.render.attachNewNode(name)
        self._part(root, "base", (0.60 * scale, 0.48 * scale, 0.32 * scale), (0.0, 0.0, 0.10 * scale),
                   (0.40, 0.38, 0.34, 1.0), (0.12, 0.12, 0.12, 1.0), (0.60, 0.58, 0.54, 1.0), 24.0, hpr=(8.0, -8.0, 14.0))
        self._part(root, "cap", (0.36 * scale, 0.32 * scale, 0.24 * scale), (0.14 * scale, -0.04 * scale, 0.28 * scale),
                   (0.52, 0.50, 0.44, 1.0), (0.16, 0.16, 0.14, 1.0), (0.72, 0.70, 0.64, 1.0), 30.0, hpr=(-12.0, 10.0, -18.0))
        self._part(root, "chip", (0.24 * scale, 0.20 * scale, 0.16 * scale), (-0.18 * scale, 0.12 * scale, 0.18 * scale),
                   (0.48, 0.46, 0.40, 1.0), (0.14, 0.14, 0.12, 1.0), (0.68, 0.66, 0.60, 1.0), 28.0, hpr=(18.0, 0.0, 24.0))
        return root

    # ------------------------------------------------------------- waypoints

    def _waypoint_colour(self, idx: int):
        if idx < self.sim.target_idx:
            return (0.45, 0.48, 0.50, 1.0)      # done
        if idx == self.sim.target_idx:
            return (0.0, 0.85, 0.65, 1.0)       # active
        return (0.86, 0.90, 0.34, 1.0)          # queued

    def _rebuild_waypoint_visuals(self) -> None:
        for marker in self.waypoint_nodes:
            marker.removeNode()
        self.waypoint_nodes.clear()
        if self._route_np is not None:
            self._route_np.removeNode()
            self._route_np = None

        for idx, waypoint in enumerate(self.sim.waypoints):
            marker = self._build_box(f"waypoint-{idx}", self._waypoint_colour(idx))
            marker.reparentTo(self.render)
            marker.setScale(0.18, 0.18, 0.9)
            marker.setPos(waypoint.x, waypoint.y, float(self.terrain.height(waypoint.x, waypoint.y)) + 0.45)
            marker.setLightOff(1)
            self.waypoint_nodes.append(marker)

        if len(self.sim.waypoints) >= 2:
            # Route line drapes over the terrain instead of cutting through hills.
            route = LineSegs("mission-route")
            route.setThickness(2.5)
            route.setColor(0.18, 0.92, 0.72, 0.8)
            first = True
            for a, b in zip(self.sim.waypoints, self.sim.waypoints[1:]):
                n = max(2, int(math.hypot(b.x - a.x, b.y - a.y) / 0.5))
                for t in np.linspace(0.0, 1.0, n):
                    x, y = a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t
                    p = (x, y, float(self.terrain.height(x, y)) + 0.10)
                    if first:
                        route.moveTo(*p)
                        first = False
                    else:
                        route.drawTo(*p)
            self._route_np = self.render.attachNewNode(route.create())
            self._route_np.setLightOff(1)
            self._route_np.setTransparency(TransparencyAttrib.MAlpha)

    def _refresh_waypoints(self) -> None:
        for idx, node in enumerate(self.waypoint_nodes):
            node.setColor(*self._waypoint_colour(idx))

    # ------------------------------------------------------------- HUD

    def _build_hud(self) -> None:
        self.hud = OnscreenText(
            text="", parent=self.a2dTopLeft, align=TextNode.ALeft, pos=(0.05, -0.07), scale=0.045,
            fg=(0.98, 0.98, 0.98, 1.0), mayChange=True, shadow=(0.05, 0.05, 0.05, 0.9),
        )
        self.help = OnscreenText(
            text="Click add waypoint   W/S throttle   A/D steer   M manual   1 auto   2 mission   C camera   Space pause   R reset   Q quit",
            parent=self.a2dBottomLeft, align=TextNode.ALeft, pos=(0.05, 0.06), scale=0.04,
            fg=(0.94, 0.94, 0.94, 1.0), mayChange=False, shadow=(0.05, 0.05, 0.05, 0.9),
        )
        self.banner = OnscreenText(
            text="", pos=(0.0, 0.55), scale=0.08, fg=(0.85, 1.0, 0.92, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.9), mayChange=True,
        )

    def _update_hud(self) -> None:
        sim, s, p = self.sim, self.sim.state, self.sim.params
        target = sim.current_waypoint()
        distance, bearing = sim.target_geometry()
        mode = "manual" if sim.manual_mode else self.control_mode
        self.hud.setText(
            f"Target: {target.name if target else 'none'}   dist {distance:.1f} m   bearing {bearing:+.0f} deg\n"
            f"Pose: x={s.x:+.2f}  y={s.y:+.2f}  z={s.z:+.2f} m   hdg {s.heading_deg:+.0f} deg\n"
            f"Speed {s.speed:.2f} m/s   yaw {s.yaw_rate_dps:+.0f} deg/s   pitch {s.pitch_deg:+.1f}   roll {s.roll_deg:+.1f}\n"
            f"Treads L {s.left_track_mps:+.2f} / R {s.right_track_mps:+.2f} m/s   "
            f"motor {p.track_to_rpm(s.left_track_mps):+.0f} / {p.track_to_rpm(s.right_track_mps):+.0f} rpm\n"
            f"Current L {sim.left_current_a:.1f} / R {sim.right_current_a:.1f} A   "
            f"power {sim.power_w:.0f} W   energy {sim.energy_j / 3600.0:.3f} Wh"
            f"{'   MOTOR LIMIT' if sim.motor_limited else ''}{'   TRACTION LIMIT' if sim.traction_limited else ''}\n"
            f"{self._ramp_hud_line()}"
            f"Mode: {mode}   Terrain: {self.terrain_mode}   Mission: {'on' if sim.mission_mode else 'off'}"
            f"   Phase: {sim.mission_phase}   Route {min(sim.target_idx, len(sim.waypoints))}/{len(sim.waypoints)}\n"
            f"Nearest obstacle: {self._nearest_obstacle_text}   Collisions: {sim.collisions}   "
            f"t = {sim.sim_time_s:.1f} s   Camera: {self._camera_mode}{'   PAUSED' if self.paused else ''}"
        )
        if sim.finished and not sim.manual_mode and self.terrain_mode == "testyard":
            climbed = len(sim.waypoints) // 2 - len(sim.skipped_waypoints)
            fail = f"  Failed: {', '.join(sim.skipped_waypoints)}" if sim.skipped_waypoints else ""
            self.banner.setText(f"Test yard done: {climbed}/{len(sim.waypoints) // 2} ramps.{fail}\n"
                                f"Max pitch {sim.max_abs_pitch_deg:.1f} deg, {sim.energy_j / 3600.0:.2f} Wh")
        elif sim.finished and not sim.manual_mode:
            self.banner.setText(
                f"Mission complete: {len(sim.samples_completed)} samples, {sim.distance_m:.1f} m, "
                f"{sim.energy_j / 3600.0:.2f} Wh\nClick the ground to add another site"
            )
        else:
            self.banner.setText("")

    def _ramp_hud_line(self) -> str:
        if self.terrain_mode != "testyard":
            return ""
        s = self.sim.state
        angle = float(self.terrain.ramp_angle(s.x, s.y))
        on = f"on {angle:g} deg ramp" if float(self.terrain.height(s.x, s.y)) > 0.02 else "on the pad"
        failed = f"   failed: {', '.join(self.sim.skipped_waypoints)}" if self.sim.skipped_waypoints else ""
        return f"Test yard: {on}   max pitch so far {self.sim.max_abs_pitch_deg:.1f} deg{failed}\n"

    def _scan_obstacles(self) -> None:
        s = self.sim.state
        nearest_name, nearest_distance = None, float("inf")
        for obs in self.sim.obstacles:
            dx, dy = obs.x - s.x, obs.y - s.y
            distance = math.hypot(dx, dy) - obs.radius_m
            bearing = abs(((math.degrees(math.atan2(dy, dx)) - s.heading_deg) + 180.0) % 360.0 - 180.0)
            visible = distance <= 11.0 and bearing <= 72.0
            node = self._obstacle_nodes[obs.name]
            if visible:
                self._detected_obstacles.add(obs.name)
                node.setColorScale(1.15, 1.15, 0.85, 1.0)
                if distance < nearest_distance:
                    nearest_name, nearest_distance = obs.name, distance
            else:
                node.clearColorScale()
        self._nearest_obstacle_text = "none" if nearest_name is None else f"{nearest_name} @ {nearest_distance:.1f} m"

    # ------------------------------------------------------------- per frame

    def _sync_scene(self, frame_dt: float) -> None:
        sim, s = self.sim, self.sim.state
        h, p, r = math.radians(s.heading_deg), math.radians(s.pitch_deg), math.radians(s.roll_deg)

        # Orient from forward + up vectors (heading, then pitch about the
        # rover's own lateral axis, then roll about its own forward axis).
        forward = LVector3f(math.cos(h) * math.cos(p), math.sin(h) * math.cos(p), math.sin(p))
        left = LVector3f(-math.sin(h) * math.cos(r), math.cos(h) * math.cos(r), math.sin(r))
        up = forward.cross(left)
        up.normalize()
        pos = Point3(s.x, s.y, s.z)
        self.rover_np.setPos(pos)
        self.rover_np.lookAt(pos + forward, up)

        # Treads: each side scrolls with its own surface speed, so pivot
        # turns show one tread forward and one back.
        dist = {"left": sim.left_track_dist_m, "right": sim.right_track_dist_m}
        for node, base_x, is_top, side in self._tread_nodes:
            shift = dist[side] if is_top else -dist[side]
            node.setX(((base_x + shift + 0.9) % 1.8) - 0.9)

        self._drill_stage_np.setZ(self._drill_rest_z - DRILL_TRAVEL_M * sim.drill_lift)
        self._drill_bit_np.setH(sim.drill_spin_deg)
        self._measure_stage_np.setZ(-PROBE_TRAVEL_M * sim.measure_lift)

        self._sky_np.setPos(s.x, s.y, s.z - 6.0)
        self._sun_np.setPos(s.x + 40.0, s.y + 56.0, s.z + 70.0)
        drift = sim.sim_time_s * 0.25
        for idx, cloud in enumerate(self._cloud_nodes):
            cloud.setPos(s.x - 40.0 + idx * 34.0 + drift % 30.0, s.y + 60.0 - idx * 12.0, s.z + 36.0 + idx * 4.0)

        self._update_camera(frame_dt)
        self._scan_obstacles()
        self._refresh_waypoints()
        self._update_trail()
        self._update_hud()
        self._render_frame += 1

    def _update_camera(self, frame_dt: float) -> None:
        s = self.sim.state
        h = math.radians(s.heading_deg)
        if self._camera_mode == "overhead":
            eye = LVector3f(s.x, s.y - 0.5, s.z + 22.0)
            target = LVector3f(s.x, s.y, s.z)
        elif self._camera_mode == "cockpit":
            eye = LVector3f(s.x + math.cos(h) * 0.5, s.y + math.sin(h) * 0.5, s.z + self._ride_height + 1.55)
            target = LVector3f(s.x + math.cos(h) * 7.0, s.y + math.sin(h) * 7.0, s.z + 0.6)
        elif self._camera_mode == "side":
            eye = LVector3f(s.x - math.sin(h) * 6.0, s.y + math.cos(h) * 6.0, s.z + 1.2)
            target = LVector3f(s.x, s.y, s.z + 0.7)
        else:
            eye = LVector3f(
                s.x - math.cos(h) * 9.0 + math.sin(h) * 3.0,
                s.y - math.sin(h) * 9.0 - math.cos(h) * 3.0,
                s.z + 6.5,
            )
            target = LVector3f(s.x + math.cos(h) * 1.5, s.y + math.sin(h) * 1.5, s.z + 0.6)

        if self._camera_pos is None or frame_dt <= 0.0:
            self._camera_pos, self._camera_target = eye, target
        else:
            # Frame-rate independent smoothing (same feel at 60 or 144 fps).
            alpha = 1.0 - math.exp(-6.0 * frame_dt)
            self._camera_pos += (eye - self._camera_pos) * alpha
            self._camera_target += (target - self._camera_target) * alpha
        self.camera.setPos(self._camera_pos)
        self.camera.lookAt(Point3(self._camera_target))

    def _update_trail(self) -> None:
        s = self.sim.state
        if not self._trail_points or math.hypot(s.x - self._trail_points[-1][0], s.y - self._trail_points[-1][1]) > 0.12:
            h = math.radians(s.heading_deg)
            half_gauge = self.sim.params.track_gauge_m / 2.0
            marks = []
            for sign in (1.0, -1.0):
                px, py = s.x - math.sin(h) * half_gauge * sign, s.y + math.cos(h) * half_gauge * sign
                marks.append((px, py, float(self.terrain.height(px, py)) + 0.03))
            self._trail_points.append((s.x, s.y, marks[0], marks[1]))
            if len(self._trail_points) > TRAIL_MAX_POINTS:
                del self._trail_points[0]
            self._trail_dirty = True
        if not self._trail_dirty:
            return
        if self._trail_np is not None:
            self._trail_np.removeNode()
            self._trail_np = None
        self._trail_dirty = False
        if len(self._trail_points) < 2:
            return

        root = self.render.attachNewNode("tread-marks")
        for side in (2, 3):
            segs = LineSegs()
            segs.setThickness(4.0)
            segs.setColor(0.18, 0.12, 0.06, 0.8)
            for i, point in enumerate(self._trail_points):
                segs.moveTo(*point[side]) if i == 0 else segs.drawTo(*point[side])
            root.attachNewNode(segs.create())
        root.setLightOff(1)
        root.setDepthOffset(2)
        self._trail_np = root

    def _read_manual_keys(self) -> None:
        throttle = (1.0 if self._keys["w"] else 0.0) - (1.0 if self._keys["s"] else 0.0)
        steer = (1.0 if self._keys["a"] else 0.0) - (1.0 if self._keys["d"] else 0.0)
        alpha = SIM_DT * 4.5
        self.sim.manual_throttle = _lerp(self.sim.manual_throttle, throttle, alpha)
        self.sim.manual_steer = _lerp(self.sim.manual_steer, steer, alpha)

    def _update_task(self, task):
        if self._quit_requested:
            self.userExit()
            return task.done

        frame_dt = min(ClockObject.getGlobalClock().getDt(), 0.1)
        if not self.paused:
            self._accumulator += frame_dt
            substeps = 0
            while self._accumulator >= SIM_DT and substeps < MAX_SUBSTEPS:
                if self.sim.manual_mode:
                    self._read_manual_keys()
                done = self.sim.step(SIM_DT)
                self._accumulator -= SIM_DT
                substeps += 1
                if done:
                    self._accumulator = 0.0
                    break
                if self.max_steps is not None and self.sim.step_idx >= self.max_steps:
                    self._quit_requested = True
                    break
            if substeps == MAX_SUBSTEPS:
                self._accumulator = 0.0  # drop backlog instead of spiralling

        self.finished = self.sim.finished
        self._sync_scene(frame_dt)

        if self._snapshot_path and not self._snapshot_taken and self._render_frame >= self._snapshot_after_frames:
            self.capture_snapshot(self._snapshot_path)
            if self._snapshot_exit:
                self._quit_requested = True

        if self.finished and self.exit_on_complete:
            self._quit_requested = True
        return task.cont

    # ------------------------------------------------------------- output

    def capture_snapshot(self, path: Optional[str] = None) -> bool:
        output_path = path or self._snapshot_path or "rover_snapshot.png"
        if self.win is None:
            log.warning("Snapshot skipped because no window is available")
            return False
        self.graphicsEngine.renderFrame()
        saved = self.win.saveScreenshot(Filename.fromOsSpecific(output_path))
        if saved:
            log.info("Snapshot saved -> %s", output_path)
            self._snapshot_taken = True
        else:
            log.warning("Snapshot capture failed -> %s", output_path)
        return bool(saved)

    def save_path_csv(self, path: str = "sim_rover_3d_path.csv") -> None:
        self.sim.log.save_csv(path)


def run_rover_3d_demo(
    waypoints: Optional[Sequence[Waypoint3D]] = None,
    show_video: bool = True,
    save_csv: bool = True,
    delay_ms: int = 40,
    target_fps: float = 90.0,
    max_steps: Optional[int] = None,
    waypoint_tolerance_m: float = 0.45,
    control_mode: str = DEFAULT_CONTROL,
    terrain_mode: str = DEFAULT_TERRAIN,
    display_resolution=(2560, 1440),
    csv_path: str = "sim_rover_3d_path.csv",
    run_label: str = "demo",
    start_manual: bool = False,
    snapshot_path: Optional[str] = None,
    snapshot_exit: bool = False,
    params: Optional[RoverParams] = None,
    mission_mode: Optional[bool] = None,
    exit_on_complete: bool = False,
    camera_mode: str = "follow",
) -> MotionLog:
    """Same signature as before (plus a few optional extras).

    show_video=False now really runs without a window (headless core).
    delay_ms is ignored (kept for compatibility).
    """
    if not show_video:
        sim = run_headless(
            waypoints=waypoints, terrain_mode=terrain_mode, control_mode=control_mode,
            mission_mode=mission_mode, csv_path=csv_path if save_csv else None,
            run_label=run_label, params=params,
            max_time_s=(max_steps * SIM_DT) if max_steps else 1800.0,
        )
        print_summary(sim.summary())
        return sim.log

    app = Rover3DPandaApp(
        waypoints=waypoints,
        waypoint_tolerance_m=waypoint_tolerance_m,
        control_mode=control_mode,
        terrain_mode=terrain_mode,
        run_label=run_label,
        max_steps=max_steps,
        display_resolution=display_resolution,
        target_fps=target_fps,
        params=params,
        mission_mode=mission_mode,
        exit_on_complete=exit_on_complete,
    )
    app.sim.manual_mode = start_manual
    app._camera_mode = camera_mode
    app._snapshot_path = snapshot_path
    app._snapshot_exit = snapshot_exit
    log.info("Starting Panda3D rover demo with %d waypoints", len(app.sim.waypoints))
    try:
        app.run()
    except SystemExit:
        pass
    finally:
        if save_csv:
            app.save_path_csv(csv_path)
    print_summary(app.sim.summary())
    return app.sim.log


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
    parser = argparse.ArgumentParser(description="DIRT rover 3D simulation (Panda3D)")
    add_common_args(parser)
    parser.add_argument("--headless", "--no-gui", dest="headless", action="store_true",
                        help="Run without a window (fast, same physics)")
    parser.add_argument("--iters", type=int, default=0, help="Max physics steps; 0 = run until done/quit")
    parser.add_argument("--speed", type=int, default=40, help="Legacy frame delay ms (ignored)")
    parser.add_argument("--fps", type=float, default=90.0, help="Target render rate")
    parser.add_argument("--resolution", type=str, default="2560x1440", help="Window size, e.g. 1920x1080")
    parser.add_argument("--manual", action="store_true", help="Start in manual driving mode")
    parser.add_argument("--camera", choices=["follow", "overhead", "cockpit", "side"], default="follow")
    parser.add_argument("--snapshot", type=str, default=None, help="Save a PNG after a few frames and exit")
    parser.add_argument("--exit-on-complete", action="store_true", help="Close the window when the mission ends")
    args = parser.parse_args()

    run_rover_3d_demo(
        waypoints=waypoints_from_args(args),
        show_video=not args.headless,
        save_csv=True,
        target_fps=args.fps,
        max_steps=None if args.iters <= 0 else args.iters,
        control_mode=args.control_mode,
        terrain_mode=args.terrain_mode,
        display_resolution=normalize_resolution(args.resolution),
        csv_path=args.csv,
        run_label=args.label,
        start_manual=args.manual,
        snapshot_path=args.snapshot,
        snapshot_exit=bool(args.snapshot),
        params=params_from_args(args),
        mission_mode=False if args.no_mission else None,
        exit_on_complete=args.exit_on_complete,
        camera_mode=args.camera,
    )


if __name__ == "__main__":
    main()