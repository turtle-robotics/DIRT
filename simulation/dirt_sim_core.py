"""
DIRT Bot - rover simulation core (no graphics)
==============================================

Everything that decides where the rover goes lives here: terrain, obstacles,
the skid-steer (tank tread) drive model, the REV HD Hex motor model, the
waypoint controller, the drill/measure mission state machine, and CSV logging.

It has no Panda3D dependency, so it runs anywhere numpy runs, including a
batch job on the TAMU HPRC Grace cluster:

    python dirt_sim_core.py                         # one headless run
    python dirt_sim_core.py --batch --out runs/     # every terrain x control combo
    python dirt_sim_core.py --random-waypoints 6 --seed 3

rover3d_panda.py draws this same simulation in 3D.

Units: metres, seconds, radians internally (degrees only at the edges: logs,
HUD, CLI). Heading 0 = +X, counter-clockwise positive.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("dirt-sim")


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------

# max_speed_mps is clamped to what the motors can actually do (see
# RoverParams.max_track_speed_mps), so "aggressive" may be motor-limited.
# steering_gain is yaw-rate command per radian of heading error (1/s).
CONTROL_PROFILES: Dict[str, Dict[str, float]] = {
    "smooth": {"max_speed_mps": 0.55, "steering_gain": 1.35, "accel_mps2": 0.28},
    "pursuit": {"max_speed_mps": 0.75, "steering_gain": 2.00, "accel_mps2": 0.45},
    "aggressive": {"max_speed_mps": 1.00, "steering_gain": 2.80, "accel_mps2": 0.62},
}

TERRAIN_PROFILES: Tuple[str, ...] = ("farmland", "grassland", "rolling", "dunes", "rocky", "testyard")
DEFAULT_TERRAIN = "farmland"
DEFAULT_CONTROL = "pursuit"

# Ground surface per terrain: (rolling resistance coeff, tread traction coeff).
# Traction coeff = max tractive force / normal load; atan(mu) is roughly the
# steepest slope the treads can grip. Ballpark values for rubber treads,
# override with --traction / --rolling-resistance once you've measured them.
SURFACE: Dict[str, Tuple[float, float]] = {
    "farmland": (0.08, 0.65),   # tilled soil
    "grassland": (0.06, 0.70),
    "rolling": (0.07, 0.70),
    "dunes": (0.12, 0.55),      # loose sand: draggy and slippery
    "rocky": (0.06, 0.75),
    "testyard": (0.04, 0.80),   # plywood / concrete ramps
}

# Test yard layout: one ramp lane per angle, ramps run along X.
TESTYARD_ANGLES_DEG: Tuple[float, ...] = (5, 10, 15, 20, 25, 30, 35, 40)
TESTYARD_LANE_SPACING_M = 5.0
TESTYARD_SLOPE_LEN_M = 4.0     # horizontal run of each slope face
TESTYARD_TOP_LEN_M = 3.0       # flat deck on top
TESTYARD_HALF_WIDTH_M = 1.8    # rover is ~2.2 m wide over the treads
TESTYARD_WALL_M = 0.25         # side wall falloff


def testyard_lanes() -> List[Tuple[float, float]]:
    """(angle_deg, lane centre y) for each ramp, gentlest first."""
    n = len(TESTYARD_ANGLES_DEG)
    return [(a, (i - (n - 1) / 2.0) * TESTYARD_LANE_SPACING_M) for i, a in enumerate(TESTYARD_ANGLES_DEG)]

GRAVITY = 9.81


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------
# Hardware model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MotorSpec:
    """REV HD Hex Motor (REV-41-1291) behind a 40:1 spur gearbox.

    Numbers are from REV's published spec. Gearbox efficiency is not
    published, so 0.80 is an assumption for a two-stage spur box.
    """

    voltage_v: float = 12.0
    free_rpm: float = 6000.0
    stall_torque_nm: float = 0.105
    stall_current_a: float = 8.5
    free_current_a: float = 0.4
    gear_ratio: float = 40.0
    gearbox_efficiency: float = 0.80

    @property
    def free_rad_s(self) -> float:
        return self.free_rpm * 2.0 * math.pi / 60.0

    @property
    def resistance_ohm(self) -> float:
        return self.voltage_v / self.stall_current_a

    @property
    def kt_nm_per_a(self) -> float:
        return self.stall_torque_nm / (self.stall_current_a - self.free_current_a)

    @property
    def ke_v_s_per_rad(self) -> float:
        return (self.voltage_v - self.free_current_a * self.resistance_ohm) / self.free_rad_s

    @property
    def output_free_rpm(self) -> float:
        return self.free_rpm / self.gear_ratio

    @property
    def output_stall_torque_nm(self) -> float:
        return self.stall_torque_nm * self.gear_ratio * self.gearbox_efficiency

    def available_output_torque(self, output_rad_s: float, braking: bool = False) -> float:
        """Max output-shaft torque at full voltage for a given output speed.

        Motoring follows the usual linear torque-speed line. Braking (torque
        opposing the motion, e.g. holding back on a downhill) gets the driver's
        full reverse voltage on top of the back-EMF, so it is *stronger* with
        speed, not weaker; capped at 2x stall to stay sane.
        """
        motor_rad_s = abs(output_rad_s) * self.gear_ratio
        if braking:
            factor = min(2.0, 1.0 + motor_rad_s / self.free_rad_s)
        else:
            factor = max(0.0, 1.0 - motor_rad_s / self.free_rad_s)
        return self.stall_torque_nm * factor * self.gear_ratio * self.gearbox_efficiency

    def current_for_output_torque(self, output_torque_nm: float) -> float:
        motor_torque = abs(output_torque_nm) / (self.gear_ratio * self.gearbox_efficiency)
        return self.free_current_a + motor_torque / self.kt_nm_per_a

    def electrical_power(self, current_a: float, output_rad_s: float) -> float:
        """Copper loss plus back-EMF power. Always >= 0 (no regen assumed)."""
        motor_rad_s = abs(output_rad_s) * self.gear_ratio
        return current_a * current_a * self.resistance_ohm + self.ke_v_s_per_rad * motor_rad_s * current_a


@dataclass
class RoverParams:
    """Physical parameters for the full rover.

    PLACEHOLDERS (update from CAD / the BOM once they are known):
    mass_kg, sprocket_radius_m. track_gauge_m and track_length_m are set to
    match the 3D model in rover3d_panda.py so the treads line up visually.
    """

    mass_kg: float = 9.0
    track_gauge_m: float = 1.88          # centre-to-centre spacing of the two treads
    track_length_m: float = 1.92         # ground-contact length of each tread
    sprocket_radius_m: float = 0.05      # drive sprocket pitch radius
    rolling_resistance: Optional[float] = None  # None = use the terrain's SURFACE value
    traction_coeff: Optional[float] = None      # None = use the terrain's SURFACE value
    turn_resistance: float = 0.45        # lateral friction coeff for skid steering
    turn_slip: float = 0.25              # fraction of commanded yaw lost to track skid
    base_power_w: float = 8.0            # MCU, Pi, sensors, spectrometer idle
    footprint_radius_m: float = 1.15     # collision radius for the whole rover
    motor: MotorSpec = field(default_factory=MotorSpec)

    @property
    def max_track_speed_mps(self) -> float:
        return self.motor.output_free_rpm * 2.0 * math.pi / 60.0 * self.sprocket_radius_m

    @property
    def max_yaw_rate_rad_s(self) -> float:
        return 2.0 * self.max_track_speed_mps / self.track_gauge_m * (1.0 - self.turn_slip)

    def track_to_rpm(self, track_mps: float) -> float:
        return track_mps / self.sprocket_radius_m * 60.0 / (2.0 * math.pi)


# --------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------

class TerrainField:
    """Analytic height field. height() accepts scalars or numpy arrays."""

    def __init__(self, span_m: float = 48.0, terrain_mode: str = DEFAULT_TERRAIN):
        self.span_m = span_m
        self.terrain_mode = terrain_mode if terrain_mode in TERRAIN_PROFILES else DEFAULT_TERRAIN

    # Crop rows run along X. Ridge crests sit where sin(y * FURROW_K) == 1.
    FURROW_K = 5.4
    FURROW_AMP = 0.035

    @staticmethod
    def _ramp_profile(x):
        """0..1 along X: up-slope, flat deck, down-slope, centred on x = 0."""
        start = TESTYARD_TOP_LEN_M / 2.0 + TESTYARD_SLOPE_LEN_M
        up = np.clip((x + start) / TESTYARD_SLOPE_LEN_M, 0.0, 1.0)
        down = np.clip((start - x) / TESTYARD_SLOPE_LEN_M, 0.0, 1.0)
        return np.minimum(up, down)

    def _testyard(self, x, y, want_angle: bool = False):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        along = self._ramp_profile(x)
        h = np.zeros(np.broadcast(x, y).shape)
        angle = np.zeros_like(h)
        for angle_deg, yc in testyard_lanes():
            lateral = np.clip((TESTYARD_HALF_WIDTH_M + TESTYARD_WALL_M - np.abs(y - yc)) / TESTYARD_WALL_M, 0.0, 1.0)
            lane_h = TESTYARD_SLOPE_LEN_M * math.tan(math.radians(angle_deg)) * along * lateral
            h = np.maximum(h, lane_h)
            angle = np.where(lateral * along > 0.0, angle_deg, angle)
        return angle if want_angle else h

    def ramp_angle(self, x, y):
        """Nominal ramp angle (deg) at (x, y) in the test yard, else 0."""
        if self.terrain_mode != "testyard":
            return np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape)
        return self._testyard(x, y, want_angle=True)

    def height(self, x, y):
        mode = self.terrain_mode
        if mode == "testyard":
            return self._testyard(x, y)
        if mode == "farmland":
            broad = 0.16 * np.sin(x / 7.5) + 0.11 * np.cos(y / 9.0) + 0.05 * np.sin((x + y) / 6.0)
            furrows = self.FURROW_AMP * np.sin(y * self.FURROW_K)
            return broad + furrows
        if mode == "grassland":
            base = 0.015 * np.sin(x / 6.8) + 0.012 * np.cos(y / 7.6)
            brush = 0.008 * np.sin((x + y) / 5.8)
            track = 0.006 * np.cos((x - 0.6 * y) / 6.5)
            return base + brush + track
        if mode == "dunes":
            return 0.55 * np.sin(x / 2.4) + 0.25 * np.sin((x + y) / 3.6) + 0.14 * np.cos(y / 4.4)
        if mode == "rocky":
            return (
                0.35 * np.sin(x / 1.8)
                + 0.25 * np.cos(y / 2.2)
                + 0.12 * np.sin((x * 1.7 + y * 1.3) / 1.6)
                + 0.08 * np.cos(np.hypot(x, y) / 1.4)
            )
        # rolling
        return (
            0.45 * np.sin(x / 2.9)
            + 0.35 * np.cos(y / 3.4)
            + 0.18 * np.sin((x + y) / 4.5)
            + 0.08 * np.cos(np.hypot(x, y) / 2.7)
        )

    def gradient(self, x, y, delta: float = 0.15):
        dx = (self.height(x + delta, y) - self.height(x - delta, y)) / (2.0 * delta)
        dy = (self.height(x, y + delta) - self.height(x, y - delta)) / (2.0 * delta)
        return dx, dy

    def height_and_gradient_grid(self, xs: np.ndarray, ys: np.ndarray, delta: float = 0.05):
        gx, gy = np.meshgrid(xs, ys)
        z = self.height(gx, gy)
        sx, sy = self.gradient(gx, gy, delta)
        return gx, gy, z, sx, sy


@dataclass(frozen=True)
class Obstacle:
    name: str
    kind: str  # "tree" or "rock"
    x: float
    y: float
    scale: float

    @property
    def radius_m(self) -> float:
        # Footprint of the canopy/trunk (tree) or the boulder (rock) at ground level.
        return (0.40 if self.kind == "tree" else 0.32) * self.scale


DEFAULT_OBSTACLES: Tuple[Obstacle, ...] = (
    Obstacle("tree-1", "tree", -4.5, -0.5, 0.95),
    Obstacle("tree-2", "tree", 2.8, 4.0, 1.05),
    Obstacle("tree-3", "tree", 6.8, -3.8, 0.90),
    Obstacle("tree-4", "tree", -18.0, -15.0, 1.25),
    Obstacle("tree-5", "tree", -14.0, 14.0, 1.10),
    Obstacle("tree-6", "tree", -2.0, 18.0, 1.35),
    Obstacle("tree-7", "tree", 11.0, 17.0, 1.20),
    Obstacle("tree-8", "tree", 19.0, 10.0, 1.30),
    Obstacle("tree-9", "tree", 18.0, -12.0, 1.15),
    Obstacle("tree-10", "tree", -18.0, 7.0, 1.05),
    Obstacle("tree-11", "tree", 12.0, -20.0, 1.30),
    Obstacle("rock-1", "rock", -1.2, 2.8, 0.55),
    Obstacle("rock-2", "rock", 4.9, -1.6, 0.45),
    Obstacle("rock-3", "rock", -6.2, 5.2, 0.60),
)


@dataclass
class Waypoint3D:
    name: str
    x: float
    y: float


START_POSE = (-8.0, -7.0, math.radians(25.0))

# Distance along the body from the drill bit (front) to the measurement probe
# (rear). Matches the 3D model in rover3d_panda.py; update from CAD.
DRILL_TO_PROBE_M = 2.20


TESTYARD_RUNUP_X = 9.0  # lane entry/exit points, clear of the 5.5 m ramp ends


def start_pose_for(terrain_mode: str) -> Tuple[float, float, float]:
    if terrain_mode == "testyard":
        return (-TESTYARD_RUNUP_X - 1.0, testyard_lanes()[0][1], 0.0)
    return START_POSE


def obstacles_for(terrain_mode: str) -> Tuple[Obstacle, ...]:
    return () if terrain_mode == "testyard" else DEFAULT_OBSTACLES


def default_waypoints(terrain_mode: str = DEFAULT_TERRAIN) -> List[Waypoint3D]:
    if terrain_mode == "testyard":
        # Serpentine over every ramp, gentlest first.
        points = []
        for i, (angle, yc) in enumerate(testyard_lanes()):
            entry, exit_ = (-TESTYARD_RUNUP_X, TESTYARD_RUNUP_X) if i % 2 == 0 else (TESTYARD_RUNUP_X, -TESTYARD_RUNUP_X)
            points.append(Waypoint3D(f"{angle:g} deg in", entry, yc))
            points.append(Waypoint3D(f"{angle:g} deg out", exit_, yc))
        return points
    return [
        Waypoint3D("Start", -8.0, -7.0),
        Waypoint3D("Sample A", -1.5, -2.5),
        Waypoint3D("Sample B", 4.0, 1.5),
        Waypoint3D("Sample C", 8.0, 6.0),
    ]


def random_waypoints(
    count: int,
    seed: int,
    span_m: float = 48.0,
    obstacles: Sequence[Obstacle] = DEFAULT_OBSTACLES,
    clearance_m: float = 2.0,
) -> List[Waypoint3D]:
    """Random sample sites that keep the rover footprint clear of obstacles."""
    rng = random.Random(seed)
    half = span_m / 2.0 - 3.0
    points = [Waypoint3D("Start", START_POSE[0], START_POSE[1])]
    tries = 0
    while len(points) < count + 1 and tries < 10000:
        tries += 1
        x, y = rng.uniform(-half, half), rng.uniform(-half, half)
        if any(math.hypot(x - o.x, y - o.y) < o.radius_m + clearance_m for o in obstacles):
            continue
        if math.hypot(x - points[-1].x, y - points[-1].y) < 3.0:
            continue
        points.append(Waypoint3D(f"Sample {len(points)}", x, y))
    return points


def load_waypoints_csv(path: str) -> List[Waypoint3D]:
    with open(path, newline="") as handle:
        return [Waypoint3D(row["name"], float(row["x"]), float(row["y"])) for row in csv.DictReader(handle)]


# --------------------------------------------------------------------------
# State and logging
# --------------------------------------------------------------------------

@dataclass
class RoverState:
    x: float
    y: float
    heading_deg: float
    speed: float = 0.0          # body forward speed over ground, m/s
    pitch_deg: float = 0.0      # nose up positive
    roll_deg: float = 0.0       # left side up positive
    z: float = 0.0
    yaw_rate_dps: float = 0.0
    left_track_mps: float = 0.0   # commanded/actual tread surface speed
    right_track_mps: float = 0.0


@dataclass
class RoverSample:
    # Columns from the original logger (kept, same order) ...
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
    # ... plus drive-train and mission columns.
    sim_time_s: float
    mission_phase: str
    yaw_rate_dps: float
    left_track_mps: float
    right_track_mps: float
    left_motor_rpm: float
    right_motor_rpm: float
    left_current_a: float
    right_current_a: float
    motor_limited: bool
    energy_wh: float
    collisions: int


@dataclass
class MotionLog:
    rows: List[RoverSample] = field(default_factory=list)
    every: int = 1  # keep every Nth step (1 = all)

    def save_csv(self, path: str = "sim_rover_3d_path.csv") -> None:
        if not self.rows:
            return
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        names = [f.name for f in fields(RoverSample)]
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=names)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row.__dict__)
        log.info("Path log saved -> %s (%d rows)", path, len(self.rows))


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

MISSION_PHASES = (
    "drive_to_drill", "lower_drill", "drill_spin", "raise_drill",
    "drive_to_measure", "lower_measure", "measure_hold", "raise_measure",
)


class RoverSim:
    """Fixed-timestep rover simulation. Call step(dt) repeatedly."""

    WAYPOINT_TIMEOUT_S = 150.0   # give up on a waypoint the rover cannot reach
    PIVOT_THRESHOLD_RAD = math.radians(55.0)  # turn in place above this heading error
    AVOID_RANGE_M = 2.6          # obstacle influence distance (beyond contact)

    def __init__(
        self,
        waypoints: Optional[Sequence[Waypoint3D]] = None,
        terrain_mode: str = DEFAULT_TERRAIN,
        control_mode: str = DEFAULT_CONTROL,
        terrain_span_m: float = 48.0,
        waypoint_tolerance_m: float = 0.45,
        max_speed_mps: Optional[float] = None,
        steering_gain: Optional[float] = None,
        accel_mps2: Optional[float] = None,
        params: Optional[RoverParams] = None,
        obstacles: Sequence[Obstacle] = DEFAULT_OBSTACLES,
        mission_mode: bool = True,
        run_label: str = "demo",
        log_every: int = 1,
    ):
        if control_mode not in CONTROL_PROFILES:
            log.warning("Unknown control mode %r, using %s", control_mode, DEFAULT_CONTROL)
            control_mode = DEFAULT_CONTROL
        if terrain_mode not in TERRAIN_PROFILES:
            log.warning("Unknown terrain mode %r, using %s", terrain_mode, DEFAULT_TERRAIN)
            terrain_mode = DEFAULT_TERRAIN

        self.params = params or RoverParams()
        profile = CONTROL_PROFILES[control_mode]
        self.control_mode = control_mode
        requested_speed = profile["max_speed_mps"] if max_speed_mps is None else max_speed_mps
        hw_limit = self.params.max_track_speed_mps
        if requested_speed > hw_limit:
            log.info(
                "%s profile asks for %.2f m/s but the motors top out at %.2f m/s; clamping",
                control_mode, requested_speed, hw_limit,
            )
        self.max_speed_mps = min(requested_speed, hw_limit)
        self.steering_gain = profile["steering_gain"] if steering_gain is None else steering_gain
        self.accel_mps2 = profile["accel_mps2"] if accel_mps2 is None else accel_mps2

        self.terrain_mode = terrain_mode
        self.terrain = TerrainField(span_m=terrain_span_m, terrain_mode=terrain_mode)
        self.rolling_resistance = (
            self.params.rolling_resistance
            if self.params.rolling_resistance is not None
            else SURFACE[terrain_mode][0]
        )
        self.obstacles: List[Obstacle] = list(obstacles)
        self.waypoint_tolerance_m = waypoint_tolerance_m
        self.waypoints: List[Waypoint3D] = list(waypoints or default_waypoints())
        self.run_label = run_label
        self.mission_mode = mission_mode
        self.log = MotionLog(every=max(1, int(log_every)))

        self.manual_mode = False
        self.manual_throttle = 0.0   # -1..1, set by the UI
        self.manual_steer = 0.0      # -1..1, + = turn left
        self.reset()

    # ---------------------------------------------------------------- setup

    def reset(self) -> None:
        x, y, h = START_POSE
        self.state = RoverState(x=x, y=y, heading_deg=math.degrees(h))
        self.target_idx = 0
        self.step_idx = 0
        self.sim_time_s = 0.0
        self.finished = False
        self.mission_phase = "drive_to_drill"
        self.mission_hold_timer = 0.0
        self.drill_lift = 0.0
        self.measure_lift = 0.0
        self.drill_spin_deg = 0.0
        self.measure_target: Optional[Tuple[float, float]] = None
        self.samples_completed: List[str] = []
        self.skipped_waypoints: List[str] = []
        self.waypoint_timer = 0.0
        self.collisions = 0
        self._in_contact: set = set()
        self.energy_j = 0.0
        self.distance_m = 0.0
        self.left_track_dist_m = 0.0
        self.right_track_dist_m = 0.0
        self.left_current_a = 0.0
        self.right_current_a = 0.0
        self.motor_limited = False
        self.traction_limited = False
        self.wheel_slip = 0.0
        self.power_w = self.params.base_power_w
        self.max_abs_pitch_deg = 0.0
        self.max_abs_roll_deg = 0.0
        self.log.rows.clear()
        self._update_pose_on_terrain()

    def add_waypoint(self, x: float, y: float, name: Optional[str] = None) -> Waypoint3D:
        waypoint = Waypoint3D(name or f"Field {len(self.waypoints)}", x, y)
        self.waypoints.append(waypoint)
        if self.finished:
            self.finished = False  # resume the mission toward the new point
        return waypoint

    def current_waypoint(self) -> Optional[Waypoint3D]:
        if self.target_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.target_idx]

    # ------------------------------------------------------------ terrain pose

    def _update_pose_on_terrain(self) -> None:
        """Rest the tread footprint on the terrain: sample front/back/left/right."""
        s, p, t = self.state, self.params, self.terrain
        h = math.radians(s.heading_deg)
        fx, fy = math.cos(h), math.sin(h)
        lx, ly = -fy, fx
        half_l, half_b = p.track_length_m / 2.0, p.track_gauge_m / 2.0
        h_front = float(t.height(s.x + fx * half_l, s.y + fy * half_l))
        h_back = float(t.height(s.x - fx * half_l, s.y - fy * half_l))
        h_left = float(t.height(s.x + lx * half_b, s.y + ly * half_b))
        h_right = float(t.height(s.x - lx * half_b, s.y - ly * half_b))
        s.z = 0.25 * (h_front + h_back + h_left + h_right)
        s.pitch_deg = math.degrees(math.atan2(h_front - h_back, p.track_length_m))
        s.roll_deg = math.degrees(math.atan2(h_left - h_right, p.track_gauge_m))
        self.max_abs_pitch_deg = max(self.max_abs_pitch_deg, abs(s.pitch_deg))
        self.max_abs_roll_deg = max(self.max_abs_roll_deg, abs(s.roll_deg))

    # --------------------------------------------------------------- control

    def _avoidance_heading(self, target_x: float, target_y: float) -> float:
        """Goal direction blended with obstacle repulsion + a tangential push.

        The tangential term slides the rover around an obstacle instead of
        stalling in front of it (plain repulsion has a dead spot dead-ahead).
        """
        s = self.state
        gx, gy = target_x - s.x, target_y - s.y
        goal_dist = math.hypot(gx, gy)
        if goal_dist < 1e-6:
            return math.radians(s.heading_deg)
        vx, vy = gx / goal_dist, gy / goal_dist
        for obs in self.obstacles:
            ox, oy = obs.x - s.x, obs.y - s.y
            centre = math.hypot(ox, oy)
            if centre < 1e-6:
                continue
            clearance = centre - obs.radius_m - self.params.footprint_radius_m
            if clearance > self.AVOID_RANGE_M:
                continue
            # Ignore obstacles behind us or beyond the goal.
            ux, uy = ox / centre, oy / centre
            if ux * vx + uy * vy < -0.2 or centre > goal_dist + obs.radius_m:
                continue
            strength = (self.AVOID_RANGE_M - max(clearance, 0.0)) / self.AVOID_RANGE_M
            strength = 1.6 * strength * strength
            # Tangent pointing toward whichever side the goal is on.
            tx, ty = -uy, ux
            if tx * vx + ty * vy < 0.0:
                tx, ty = -tx, -ty
            vx += strength * (-ux * 0.6 + tx)
            vy += strength * (-uy * 0.6 + ty)
        return math.atan2(vy, vx)

    def _drive_command_to_point(self, target_x: float, target_y: float, tolerance: float) -> Tuple[float, float, bool]:
        """Return (forward speed, yaw rate) commands and whether we have arrived."""
        s = self.state
        distance = math.hypot(target_x - s.x, target_y - s.y)
        if distance <= tolerance:
            return 0.0, 0.0, True

        heading = math.radians(s.heading_deg)
        desired = self._avoidance_heading(target_x, target_y)
        error = wrap_angle_rad(desired - heading)

        omega = clamp(self.steering_gain * error, -self.params.max_yaw_rate_rad_s, self.params.max_yaw_rate_rad_s)
        if abs(error) > self.PIVOT_THRESHOLD_RAD:
            v = 0.0  # tank treads: pivot in place
        else:
            v = self.max_speed_mps * clamp(distance / 2.5, 0.2, 1.0) * math.cos(error)
        return v, omega, False

    def _tracks_from_body(self, v: float, omega: float) -> Tuple[float, float]:
        """Inverse skid-steer kinematics, saturated to the motors' top speed."""
        p = self.params
        # Ask for extra differential to make up for the yaw that the skid eats.
        omega_cmd = omega / max(1e-3, 1.0 - p.turn_slip)
        left = v - omega_cmd * p.track_gauge_m / 2.0
        right = v + omega_cmd * p.track_gauge_m / 2.0
        peak = max(abs(left), abs(right))
        limit = p.max_track_speed_mps
        if peak > limit:  # scale both so the turn radius is preserved
            left *= limit / peak
            right *= limit / peak
        return left, right

    # --------------------------------------------------------------- physics

    def _apply_drivetrain(self, left_cmd: float, right_cmd: float, dt: float) -> None:
        """Slew tracks toward command, check motor torque, integrate the pose."""
        s, p, m = self.state, self.params, self.params.motor
        max_dv = self.accel_mps2 * dt

        new_left = s.left_track_mps + clamp(left_cmd - s.left_track_mps, -max_dv, max_dv)
        new_right = s.right_track_mps + clamp(right_cmd - s.right_track_mps, -max_dv, max_dv)

        # Loads. Pitch is nose-up positive, so climbing adds m*g*sin(pitch).
        pitch = math.radians(s.pitch_deg)
        a_long = ((new_left + new_right) - (s.left_track_mps + s.right_track_mps)) / (2.0 * dt)
        moving = math.tanh(abs(new_left + new_right) / 0.1)
        roll_force = p.mass_kg * GRAVITY * self.rolling_resistance * math.cos(pitch) * moving
        grade_force = p.mass_kg * GRAVITY * math.sin(pitch)
        direction = 1.0 if (new_left + new_right) >= 0.0 else -1.0
        f_long = p.mass_kg * a_long + grade_force + direction * roll_force

        # Skid-steer turning moment resistance M = mu * m * g * L / 4,
        # balanced by +/- delta force on the two treads.
        yaw_cmd = (new_right - new_left) / p.track_gauge_m
        turning = math.tanh(abs(yaw_cmd) / 0.05)
        delta = p.turn_resistance * p.mass_kg * GRAVITY * p.track_length_m / (4.0 * p.track_gauge_m) * turning
        sign = 1.0 if yaw_cmd >= 0.0 else -1.0
        f_left = f_long / 2.0 - sign * delta
        f_right = f_long / 2.0 + sign * delta

        self.motor_limited = False
        results = []
        for track_speed, force in ((new_left, f_left), (new_right, f_right)):
            sprocket_rad_s = track_speed / p.sprocket_radius_m
            torque_needed = force * p.sprocket_radius_m
            braking = track_speed * force < 0.0
            torque_avail = m.available_output_torque(sprocket_rad_s, braking=braking)
            if abs(torque_needed) > torque_avail:
                # Motor can't deliver this load at this speed: the force
                # deficit (on half the vehicle mass) eats into track speed.
                self.motor_limited = True
                deficit = (abs(torque_needed) - torque_avail) / p.sprocket_radius_m
                dv = deficit / (p.mass_kg / 2.0) * dt
                track_speed -= math.copysign(dv, force)
                torque_needed = math.copysign(torque_avail, torque_needed)
            # A geared DC motor can't be pushed past its free speed for long.
            track_speed = clamp(track_speed, -1.05 * p.max_track_speed_mps, 1.05 * p.max_track_speed_mps)
            current = m.current_for_output_torque(torque_needed)
            power = m.electrical_power(current, track_speed / p.sprocket_radius_m)
            results.append((track_speed, current, power))

        (s.left_track_mps, self.left_current_a, p_left), (s.right_track_mps, self.right_current_a, p_right) = results
        self.power_w = p.base_power_w + p_left + p_right

        # Longitudinal slip grows with grade and with how hard we are turning.
        self.wheel_slip = clamp(0.03 + 0.25 * abs(math.sin(pitch)) + 0.10 * turning, 0.0, 0.6)
        v = 0.5 * (s.left_track_mps + s.right_track_mps) * (1.0 - self.wheel_slip)
        omega = (s.right_track_mps - s.left_track_mps) / p.track_gauge_m * (1.0 - p.turn_slip)

        heading = math.radians(s.heading_deg) + omega * dt * 0.5
        dx = math.cos(heading) * v * dt
        dy = math.sin(heading) * v * dt
        s.x += dx
        s.y += dy
        s.heading_deg = wrap_angle_deg(math.degrees(math.radians(s.heading_deg) + omega * dt))
        s.speed = v
        s.yaw_rate_dps = math.degrees(omega)

        self.distance_m += math.hypot(dx, dy)
        self.left_track_dist_m += s.left_track_mps * dt
        self.right_track_dist_m += s.right_track_mps * dt
        self.energy_j += self.power_w * dt

        self._limit_to_world()
        self._resolve_collisions()
        self._update_pose_on_terrain()

    def _limit_to_world(self) -> None:
        half = self.terrain.span_m / 2.0 - self.params.footprint_radius_m
        self.state.x = clamp(self.state.x, -half, half)
        self.state.y = clamp(self.state.y, -half, half)

    def _resolve_collisions(self) -> None:
        s = self.state
        touching = set()
        for obs in self.obstacles:
            ox, oy = s.x - obs.x, s.y - obs.y
            dist = math.hypot(ox, oy)
            min_dist = obs.radius_m + self.params.footprint_radius_m
            if dist >= min_dist:
                continue
            touching.add(obs.name)
            if obs.name not in self._in_contact:
                self.collisions += 1
                log.warning("Collision with %s at t=%.1fs", obs.name, self.sim_time_s)
            if dist < 1e-6:
                ox, oy, dist = 1.0, 0.0, 1.0
            s.x = obs.x + ox / dist * min_dist
            s.y = obs.y + oy / dist * min_dist
            # Kill only the forward drive *into* the obstacle. Pivoting and
            # backing away stay free, otherwise the rover pins itself.
            h = math.radians(s.heading_deg)
            mean = 0.5 * (s.left_track_mps + s.right_track_mps)
            into = (math.cos(h) * -ox + math.sin(h) * -oy) / dist * mean
            if into > 0.0:
                s.left_track_mps -= mean
                s.right_track_mps -= mean
        self._in_contact = touching

    # --------------------------------------------------------------- stepping

    def _step_manual(self, dt: float) -> None:
        v = self.max_speed_mps * clamp(self.manual_throttle, -1.0, 1.0)
        omega = self.params.max_yaw_rate_rad_s * clamp(self.manual_steer, -1.0, 1.0)
        self._apply_drivetrain(*self._tracks_from_body(v, omega), dt)

    def _advance_waypoint(self, reached: bool) -> None:
        target = self.current_waypoint()
        if target is not None:
            if reached:
                log.info("Reached %s at t=%.1fs", target.name, self.sim_time_s)
            else:
                log.warning("Skipping %s after %.0fs without reaching it", target.name, self.waypoint_timer)
                self.skipped_waypoints.append(target.name)
        self.target_idx += 1
        self.waypoint_timer = 0.0

    def _step_autonomous(self, dt: float) -> None:
        target = self.current_waypoint()
        if target is None:
            self._apply_drivetrain(0.0, 0.0, dt)
            return
        v, omega, arrived = self._drive_command_to_point(target.x, target.y, self.waypoint_tolerance_m)
        self._apply_drivetrain(*self._tracks_from_body(v, omega), dt)
        if arrived:
            self._advance_waypoint(True)
        elif self.waypoint_timer > self.WAYPOINT_TIMEOUT_S:
            self._advance_waypoint(False)

    def _step_mission(self, dt: float) -> None:
        target = self.current_waypoint()
        if target is None:
            self._apply_drivetrain(0.0, 0.0, dt)
            return

        phase = self.mission_phase
        if phase == "drive_to_drill":
            v, omega, arrived = self._drive_command_to_point(target.x, target.y, self.waypoint_tolerance_m)
            self._apply_drivetrain(*self._tracks_from_body(v, omega), dt)
            if arrived:
                self.mission_phase = "lower_drill"
                self.mission_hold_timer = 0.0
            elif self.waypoint_timer > self.WAYPOINT_TIMEOUT_S:
                self._advance_waypoint(False)
            return

        # All the tool phases hold the rover still (and still draw base power).
        if phase != "drive_to_measure":
            self._apply_drivetrain(0.0, 0.0, dt)

        if phase == "lower_drill":
            self.drill_lift = clamp(self.drill_lift + dt * 0.9, 0.0, 1.0)
            if self.drill_lift >= 1.0:
                self.mission_phase = "drill_spin"
                self.mission_hold_timer = 0.0
        elif phase == "drill_spin":
            self.drill_spin_deg = (self.drill_spin_deg + dt * 720.0) % 360.0
            self.mission_hold_timer += dt
            self.power_w += 25.0  # drill motor load (placeholder)
            self.energy_j += 25.0 * dt
            if self.mission_hold_timer >= 2.0:
                self.mission_phase = "raise_drill"
        elif phase == "raise_drill":
            self.drill_lift = clamp(self.drill_lift - dt * 0.9, 0.0, 1.0)
            if self.drill_lift <= 0.0:
                # Creep forward so the measurement probe (rear) ends up over
                # the hole the drill (front) just made.
                h = math.radians(self.state.heading_deg)
                d = DRILL_TO_PROBE_M
                self.measure_target = (self.state.x + math.cos(h) * d, self.state.y + math.sin(h) * d)
                self.mission_phase = "drive_to_measure"
                self.mission_hold_timer = 0.0
        elif phase == "drive_to_measure":
            tx, ty = self.measure_target or (self.state.x, self.state.y)
            # Straight creep along the current heading (no avoidance: the
            # target is one body length ahead and the drill site was clear).
            s = self.state
            h = math.radians(s.heading_deg)
            ahead = (tx - s.x) * math.cos(h) + (ty - s.y) * math.sin(h)
            if ahead <= 0.05 or self.mission_hold_timer > 20.0:
                self._apply_drivetrain(0.0, 0.0, dt)
                self.mission_phase = "lower_measure"
                self.mission_hold_timer = 0.0
            else:
                self.mission_hold_timer += dt
                v = min(0.25, self.max_speed_mps) * clamp(ahead / 0.4, 0.25, 1.0)
                self._apply_drivetrain(*self._tracks_from_body(v, 0.0), dt)
        elif phase == "lower_measure":
            self.measure_lift = clamp(self.measure_lift + dt * 1.0, 0.0, 1.0)
            if self.measure_lift >= 1.0:
                self.mission_phase = "measure_hold"
                self.mission_hold_timer = 0.0
        elif phase == "measure_hold":
            self.mission_hold_timer += dt
            if self.mission_hold_timer >= 1.1:
                log.info("Measurement complete at %s", target.name)
                self.samples_completed.append(target.name)
                self.mission_phase = "raise_measure"
        elif phase == "raise_measure":
            self.measure_lift = clamp(self.measure_lift - dt * 1.0, 0.0, 1.0)
            if self.measure_lift <= 0.0:
                self._advance_waypoint(True)
                self.mission_phase = "drive_to_drill"
                self.measure_target = None
                self.mission_hold_timer = 0.0

    def step(self, dt: float) -> bool:
        """Advance one fixed step. Returns True once every waypoint is done."""
        if self.finished and not self.manual_mode:
            return True

        if self.manual_mode:
            self._step_manual(dt)
        elif self.mission_mode:
            self._step_mission(dt)
        else:
            self._step_autonomous(dt)

        self.sim_time_s += dt
        self.waypoint_timer += dt
        if not self.manual_mode and self.current_waypoint() is None and self.mission_phase == "drive_to_drill":
            self.finished = True
            self.state.left_track_mps = self.state.right_track_mps = 0.0
            self.state.speed = 0.0

        if self.step_idx % self.log.every == 0:
            self._record()
        self.step_idx += 1
        return self.finished and not self.manual_mode

    def set_mission_mode(self, enabled: bool) -> None:
        self.mission_mode = enabled
        self.mission_phase = "drive_to_drill"
        self.mission_hold_timer = 0.0
        self.drill_lift = self.measure_lift = 0.0
        self.measure_target = None

    # --------------------------------------------------------------- logging

    def target_geometry(self) -> Tuple[float, float]:
        target = self.current_waypoint()
        if target is None:
            return 0.0, 0.0
        dx, dy = target.x - self.state.x, target.y - self.state.y
        bearing = wrap_angle_deg(math.degrees(math.atan2(dy, dx)) - self.state.heading_deg)
        return math.hypot(dx, dy), bearing

    def _record(self) -> None:
        s, p = self.state, self.params
        target = self.current_waypoint()
        distance, bearing = self.target_geometry()
        curvature = abs(s.yaw_rate_dps) / max(abs(s.speed), 0.05) if abs(s.speed) > 0.01 else 0.0
        self.log.rows.append(
            RoverSample(
                step=self.step_idx,
                run_label=self.run_label,
                control_mode="manual" if self.manual_mode else self.control_mode,
                terrain_mode=self.terrain_mode,
                target=target.name if target else "done",
                x=round(s.x, 3),
                y=round(s.y, 3),
                z=round(s.z, 3),
                heading_deg=round(s.heading_deg, 2),
                speed=round(s.speed, 3),
                target_distance_m=round(distance, 3),
                target_bearing_deg=round(bearing, 2),
                terrain_height=round(float(self.terrain.height(s.x, s.y)), 3),
                pitch_deg=round(s.pitch_deg, 2),
                roll_deg=round(s.roll_deg, 2),
                wheel_slip=round(self.wheel_slip, 3),
                power_draw=round(self.power_w, 2),
                path_curvature=round(curvature, 3),
                reached=target is None,
                sim_time_s=round(self.sim_time_s, 3),
                mission_phase=self.mission_phase if self.mission_mode else "-",
                yaw_rate_dps=round(s.yaw_rate_dps, 2),
                left_track_mps=round(s.left_track_mps, 3),
                right_track_mps=round(s.right_track_mps, 3),
                left_motor_rpm=round(p.track_to_rpm(s.left_track_mps), 1),
                right_motor_rpm=round(p.track_to_rpm(s.right_track_mps), 1),
                left_current_a=round(self.left_current_a, 2),
                right_current_a=round(self.right_current_a, 2),
                motor_limited=self.motor_limited,
                energy_wh=round(self.energy_j / 3600.0, 4),
                collisions=self.collisions,
            )
        )

    def summary(self) -> Dict[str, object]:
        rows = self.log.rows
        mean_power = (self.energy_j / self.sim_time_s) if self.sim_time_s > 0 else 0.0
        peak_current = max((max(r.left_current_a, r.right_current_a) for r in rows), default=0.0)
        return {
            "run_label": self.run_label,
            "terrain_mode": self.terrain_mode,
            "control_mode": self.control_mode,
            "completed": self.finished,
            "sim_time_s": round(self.sim_time_s, 2),
            "distance_m": round(self.distance_m, 2),
            "energy_wh": round(self.energy_j / 3600.0, 3),
            "mean_power_w": round(mean_power, 2),
            "peak_motor_current_a": round(peak_current, 2),
            "max_abs_pitch_deg": round(self.max_abs_pitch_deg, 2),
            "max_abs_roll_deg": round(self.max_abs_roll_deg, 2),
            "collisions": self.collisions,
            "samples_completed": len(self.samples_completed),
            "waypoints_skipped": len(self.skipped_waypoints),
        }


# --------------------------------------------------------------------------
# Headless runners
# --------------------------------------------------------------------------

def run_headless(
    waypoints: Optional[Sequence[Waypoint3D]] = None,
    terrain_mode: str = DEFAULT_TERRAIN,
    control_mode: str = DEFAULT_CONTROL,
    mission_mode: bool = True,
    dt: float = 1.0 / 60.0,
    max_time_s: float = 1800.0,
    csv_path: Optional[str] = "sim_rover_3d_path.csv",
    run_label: str = "headless",
    params: Optional[RoverParams] = None,
    log_every: int = 1,
) -> RoverSim:
    sim = RoverSim(
        waypoints=waypoints,
        terrain_mode=terrain_mode,
        control_mode=control_mode,
        mission_mode=mission_mode,
        params=params,
        run_label=run_label,
        log_every=log_every,
    )
    max_steps = int(max_time_s / dt)
    for _ in range(max_steps):
        if sim.step(dt):
            break
    else:
        log.warning("%s: hit the %.0fs time limit before finishing", run_label, max_time_s)
    if csv_path:
        sim.log.save_csv(csv_path)
    return sim


def run_batch(
    out_dir: str,
    terrains: Sequence[str] = TERRAIN_PROFILES,
    controls: Sequence[str] = tuple(CONTROL_PROFILES),
    waypoints: Optional[Sequence[Waypoint3D]] = None,
    params: Optional[RoverParams] = None,
    dt: float = 1.0 / 60.0,
    log_every: int = 6,
    mission_mode: bool = True,
) -> List[Dict[str, object]]:
    """Run every terrain x control combination and write a summary CSV."""
    os.makedirs(out_dir, exist_ok=True)
    summaries = []
    for terrain in terrains:
        for control in controls:
            label = f"{terrain}-{control}"
            sim = run_headless(
                waypoints=waypoints,
                terrain_mode=terrain,
                control_mode=control,
                mission_mode=mission_mode,
                dt=dt,
                csv_path=os.path.join(out_dir, f"{label}.csv"),
                run_label=label,
                params=params,
                log_every=log_every,
            )
            summaries.append(sim.summary())
    summary_path = os.path.join(out_dir, "batch_summary.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    log.info("Batch summary -> %s", summary_path)
    return summaries


def print_summary(summary: Dict[str, object]) -> None:
    print("\nRover Simulation Summary")
    width = max(len(k) for k in summary)
    for key, value in summary.items():
        print(f"  {key.ljust(width)} : {value}")
    print()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """CLI flags shared by the headless runner and the 3D viewer."""
    parser.add_argument("--waypoints", type=str, default=None, help="CSV with columns name,x,y")
    parser.add_argument("--random-waypoints", type=int, default=0, metavar="N", help="Use N random sample sites")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --random-waypoints")
    parser.add_argument("--csv", type=str, default="sim_rover_3d_path.csv", help="CSV output path")
    parser.add_argument("--control-mode", choices=sorted(CONTROL_PROFILES), default=DEFAULT_CONTROL)
    parser.add_argument("--terrain-mode", choices=list(TERRAIN_PROFILES), default=DEFAULT_TERRAIN)
    parser.add_argument("--label", type=str, default="demo", help="Run label written into the CSV")
    parser.add_argument("--no-mission", action="store_true", help="Just drive the waypoints (no drill/measure)")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Physics timestep in seconds")
    parser.add_argument("--mass", type=float, default=None, help="Rover mass, kg")
    parser.add_argument("--sprocket-radius", type=float, default=None, help="Drive sprocket pitch radius, m")
    parser.add_argument("--track-gauge", type=float, default=None, help="Tread centre spacing, m")


def params_from_args(args: argparse.Namespace) -> RoverParams:
    params = RoverParams()
    if args.mass is not None:
        params.mass_kg = args.mass
    if args.sprocket_radius is not None:
        params.sprocket_radius_m = args.sprocket_radius
    if args.track_gauge is not None:
        params.track_gauge_m = args.track_gauge
    return params


def waypoints_from_args(args: argparse.Namespace) -> Optional[List[Waypoint3D]]:
    if args.waypoints:
        return load_waypoints_csv(args.waypoints)
    if args.random_waypoints > 0:
        return random_waypoints(args.random_waypoints, args.seed)
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
    parser = argparse.ArgumentParser(description="DIRT rover simulation, headless (no graphics)")
    add_common_args(parser)
    parser.add_argument("--batch", action="store_true", help="Run every terrain x control combination")
    parser.add_argument("--out", type=str, default="sim_runs", help="Output folder for --batch")
    parser.add_argument("--max-time", type=float, default=1800.0, help="Give up after this many sim seconds")
    parser.add_argument("--log-every", type=int, default=1, help="Keep every Nth step in the CSV")
    args = parser.parse_args()

    params = params_from_args(args)
    waypoints = waypoints_from_args(args)
    print(
        f"Drive: REV HD Hex 40:1 x2, {params.motor.output_free_rpm:.0f} rpm out, "
        f"sprocket r={params.sprocket_radius_m:.3f} m -> top speed {params.max_track_speed_mps:.2f} m/s"
    )

    if args.batch:
        summaries = run_batch(
            args.out, waypoints=waypoints, params=params, dt=args.dt,
            log_every=max(args.log_every, 1), mission_mode=not args.no_mission,
        )
        for summary in summaries:
            print(
                f"  {summary['run_label']:<22} done={str(summary['completed']):<5} "
                f"t={summary['sim_time_s']:>7.1f}s  E={summary['energy_wh']:>6.3f}Wh  "
                f"hits={summary['collisions']}  maxPitch={summary['max_abs_pitch_deg']:.1f}deg"
            )
        return

    sim = run_headless(
        waypoints=waypoints,
        terrain_mode=args.terrain_mode,
        control_mode=args.control_mode,
        mission_mode=not args.no_mission,
        dt=args.dt,
        max_time_s=args.max_time,
        csv_path=args.csv,
        run_label=args.label,
        params=params,
        log_every=args.log_every,
    )
    print_summary(sim.summary())


if __name__ == "__main__":
    main()
