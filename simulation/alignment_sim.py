"""
DIRT Bot – Alignment Simulation Harness
=========================================
Generates synthetic camera frames with red markers at controlled
offsets and runs the full alignment control loop – no hardware needed.

This is what the reviewer asked for: "simulations of the autonomous part".
Run it with:  python simulation/alignment_sim.py

Features:
  • Animated OpenCV window showing the virtual camera view
  • Real-time plots of x/y/rotation error vs. iteration
  • Configurable initial misalignment and noise level
  • Saves a convergence CSV for the PDR report
  • Command-line args for quick parameter sweeps
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CFG, Platform, ControlConfig, ToleranceConfig
from core.alignment_system import (
    AlignmentDetector, AlignmentData, ControlLaw,
    SimulatedMotors, MotorCommand, MotorDirection
)

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("sim")


# ── Synthetic world state ──────────────────────────────────────────────────────

@dataclass
class RobotPose:
    """Virtual position of the spectrometer box relative to the target hole."""
    x_px:     float   # Pixel offset from camera centre
    y_px:     float   # Pixel offset from camera centre
    angle_deg: float  # Rotation offset in degrees

    def apply_command(self, cmd: Optional[MotorCommand], noise: float = 1.0):
        """Update pose based on motor command + Gaussian noise."""
        if cmd is None:
            return
        gain = 0.8   # Simulated motor efficiency

        if cmd.x_dir == MotorDirection.FORWARD:
            self.x_px -= cmd.x_speed * gain * cmd.duration
        elif cmd.x_dir == MotorDirection.BACKWARD:
            self.x_px += cmd.x_speed * gain * cmd.duration

        if cmd.y_dir == MotorDirection.FORWARD:
            self.y_px -= cmd.y_speed * gain * cmd.duration
        elif cmd.y_dir == MotorDirection.BACKWARD:
            self.y_px += cmd.y_speed * gain * cmd.duration

        if cmd.z_dir == MotorDirection.FORWARD:
            self.angle_deg -= cmd.z_speed * gain * cmd.duration / 10.0
        elif cmd.z_dir == MotorDirection.BACKWARD:
            self.angle_deg += cmd.z_speed * gain * cmd.duration / 10.0

        # Gaussian noise on final pose
        self.x_px     += np.random.normal(0, noise)
        self.y_px     += np.random.normal(0, noise)
        self.angle_deg += np.random.normal(0, noise * 0.05)


# ── Synthetic frame generator ─────────────────────────────────────────────────

class SyntheticCamera:
    """
    Renders a fake camera frame with two red marker dots at the given pose offset.
    The alignment detector sees this frame exactly as it would a real frame.
    """
    W, H = CFG.camera.frame_width, CFG.camera.frame_height
    MARKER_RADIUS = 12
    MARKER_SEP    = 80   # pixels between the two markers

    def render(self, pose: RobotPose, noise_px: float = 1.0) -> np.ndarray:
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        frame[:] = (40, 35, 30)   # Dark soil-ish background

        # Draw target crosshair (the hole centre)
        cx, cy = self.W // 2, self.H // 2
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 180, 0), 1)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 180, 0), 1)
        cv2.circle(frame, (cx, cy), 4, (0, 200, 0), 1)

        # Marker positions relative to world centre + pose offset
        angle_rad = math.radians(pose.angle_deg)
        dx  = self.MARKER_SEP * math.cos(angle_rad) / 2
        dy  = self.MARKER_SEP * math.sin(angle_rad) / 2

        m1x = int(cx + pose.x_px - dx + np.random.normal(0, noise_px))
        m1y = int(cy + pose.y_px - dy + np.random.normal(0, noise_px))
        m2x = int(cx + pose.x_px + dx + np.random.normal(0, noise_px))
        m2y = int(cy + pose.y_px + dy + np.random.normal(0, noise_px))

        # Draw filled red circles (HSV-red = BGR ~(0,0,200))
        for (mx, my) in [(m1x, m1y), (m2x, m2y)]:
            cv2.circle(frame, (mx, my), self.MARKER_RADIUS, (0, 0, 200), -1)
            # Slight bright specular highlight to look realistic
            cv2.circle(frame, (mx - 3, my - 3), 3, (80, 80, 255), -1)

        # Optional Gaussian image noise
        if noise_px > 0:
            noise_img = np.random.normal(0, noise_px * 2,
                                         frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise_img,
                            0, 255).astype(np.uint8)
        return frame


# ── Convergence log ────────────────────────────────────────────────────────────

@dataclass
class ConvergenceLog:
    rows: List[dict] = field(default_factory=list)

    def record(self, iteration: int, data: AlignmentData, pose: RobotPose):
        self.rows.append({
            "iteration":   iteration,
            "x_offset_px": round(data.x_offset, 2),
            "y_offset_px": round(data.y_offset, 2),
            "rotation_deg": round(data.rotation_angle, 2),
            "confidence":  round(data.confidence, 3),
            "true_x_px":   round(pose.x_px, 2),
            "true_y_px":   round(pose.y_px, 2),
            "true_angle":  round(pose.angle_deg, 2),
            "is_aligned":  data.is_aligned,
        })

    def save_csv(self, path: str = "sim_convergence.csv"):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            w.writeheader()
            w.writerows(self.rows)
        log.info("Convergence log saved → %s", path)

    def print_summary(self):
        if not self.rows:
            print("No data recorded.")
            return
        last = self.rows[-1]
        n    = len(self.rows)
        print("\n── Simulation Summary ──────────────────────")
        print(f"  Total iterations : {n}")
        print(f"  Final X error    : {last['x_offset_px']:+.2f} px")
        print(f"  Final Y error    : {last['y_offset_px']:+.2f} px")
        print(f"  Final rotation   : {last['rotation_deg']:+.2f} °")
        print(f"  Aligned          : {last['is_aligned']}")
        print("────────────────────────────────────────────\n")


# ── Live plot overlay (pure OpenCV – no matplotlib dependency) ────────────────

class LivePlot:
    """Draws an error-vs-iteration strip chart in a separate OpenCV window."""

    def __init__(self, w: int = 640, h: int = 300, max_iter: int = 100):
        self.w, self.h   = w, h
        self.max_iter    = max_iter
        self.x_hist: List[float] = []
        self.y_hist: List[float] = []
        self.r_hist: List[float] = []

    def update(self, data: AlignmentData):
        self.x_hist.append(data.x_offset)
        self.y_hist.append(data.y_offset)
        self.r_hist.append(data.rotation_angle)

    def render(self) -> np.ndarray:
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        # Zero line
        mid = self.h // 2
        cv2.line(canvas, (0, mid), (self.w, mid), (60, 60, 60), 1)

        max_val = 150.0   # px – clamp display range
        n = len(self.x_hist)
        if n < 2:
            return canvas

        def _to_y(v):
            return int(mid - (v / max_val) * (self.h * 0.45))

        step = max(1, self.w // max(n, 1))

        for series, colour in [
            (self.x_hist, (80, 80, 255)),    # X = red-ish
            (self.y_hist, (80, 255, 80)),    # Y = green
            (self.r_hist, (255, 180, 80)),   # R = amber
        ]:
            pts = [(_to_y(v), i) for i, v in enumerate(series)]
            for idx in range(len(pts) - 1):
                y1, i1 = pts[idx]
                y2, i2 = pts[idx + 1]
                x1 = int(i1 / (n - 1) * (self.w - 1)) if n > 1 else 0
                x2 = int(i2 / (n - 1) * (self.w - 1)) if n > 1 else 0
                cv2.line(canvas, (x1, y1), (x2, y2), colour, 2)

        # Legend
        for i, (label, colour) in enumerate([
            ("X error (px)", (80, 80, 255)),
            ("Y error (px)", (80, 255, 80)),
            ("Rotation (°)", (255, 180, 80)),
        ]):
            cv2.putText(canvas, label, (10, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

        # Tolerance band
        tol_y_pos = _to_y(CFG.tolerance.x_pixels)
        tol_y_neg = _to_y(-CFG.tolerance.x_pixels)
        cv2.rectangle(canvas, (0, tol_y_neg), (self.w, tol_y_pos),
                      (0, 80, 0), 1)
        return canvas


# ── Main simulation loop ───────────────────────────────────────────────────────

def run_simulation(
    init_x:   float = 80.0,
    init_y:   float = -60.0,
    init_rot: float = 15.0,
    noise_px: float = 1.5,
    max_iter: int   = 80,
    show_video: bool = True,
    save_csv: bool  = True,
    delay_ms: int   = 50,
) -> ConvergenceLog:
    """
    Run one simulation scenario and return the convergence log.
    """
    pose     = RobotPose(init_x, init_y, init_rot)
    camera   = SyntheticCamera()
    detector = AlignmentDetector()
    control  = ControlLaw()
    log_data = ConvergenceLog()
    plot     = LivePlot(max_iter=max_iter) if show_video else None

    log.info("Simulation start: x=%+.0fpx  y=%+.0fpx  rot=%+.1f°  noise=%.1fpx",
             init_x, init_y, init_rot, noise_px)

    for iteration in range(max_iter):
        # Render synthetic frame at current pose
        frame = camera.render(pose, noise_px=noise_px)

        # Run detector (identical code path to real hardware)
        data  = detector.detect(frame)

        if data is not None:
            log_data.record(iteration, data, pose)
            if plot:
                plot.update(data)
            log.info("Iter %3d: X=%+6.1f  Y=%+6.1f  R=%+5.1f°  conf=%.2f  %s",
                     iteration, data.x_offset, data.y_offset,
                     data.rotation_angle, data.confidence,
                     "✓ ALIGNED" if data.is_aligned else "")

            if data.is_aligned:
                log.info("Alignment achieved in %d iterations!", iteration + 1)
                if show_video:
                    annotated = detector.annotate(frame, data)
                    cv2.putText(annotated, f"ALIGNED in {iteration+1} iterations",
                                (10, frame.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("DIRT Sim – Camera", annotated)
                    cv2.waitKey(1500)
                break

            # Compute and apply motor command
            cmd = control.compute(data)
            pose.apply_command(cmd, noise=noise_px * 0.5)

        if show_video:
            annotated = detector.annotate(frame, data)
            cv2.putText(annotated, f"Iteration: {iteration}",
                        (frame.shape[1] - 130, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            cv2.imshow("DIRT Sim – Camera", annotated)
            if plot:
                cv2.imshow("DIRT Sim – Error plot", plot.render())
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                log.info("Quit by user")
                break

    if show_video:
        cv2.destroyAllWindows()

    log_data.print_summary()
    if save_csv and log_data.rows:
        log_data.save_csv("sim_convergence.csv")

    return log_data


def run_parameter_sweep():
    """
    Sweep over initial offsets and noise levels to characterise system robustness.
    Outputs a CSV table – good for PDR slides.
    """
    results = []
    configs = [
        (40,  -30,  5,  0.5,  "easy"),
        (80,  -60,  15, 1.5,  "nominal"),
        (120, -100, 25, 3.0,  "hard"),
        (150,  120, 35, 5.0,  "extreme"),
    ]
    print("\n── Parameter Sweep ─────────────────────────────────────────────")
    print(f"{ 'Scenario':<12 } {'Init X':>7} {'Init Y':>7} {'Init R':>7} "
          f"{'Noise':>6} {'Iters':>6} {'Success':>8}")
    print("─" * 60)

    for (ix, iy, ir, noise, label) in configs:
        clog = run_simulation(
            init_x=ix, init_y=iy, init_rot=ir,
            noise_px=noise, max_iter=100,
            show_video=False, save_csv=False
        )
        n_iters = len(clog.rows)
        success = clog.rows[-1]["is_aligned"] if clog.rows else False
        print(f"{label:<12} {ix:>7.0f} {iy:>7.0f} {ir:>7.0f} "
              f"{noise:>6.1f} {n_iters:>6} {'✓' if success else '✗':>8}")
        results.append({
            "scenario": label, "init_x": ix, "init_y": iy,
            "init_rot": ir, "noise": noise, "iterations": n_iters,
            "success": success,
        })

    print("────────────────────────────────────────────────────────────\n")

    with open("sim_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print("Sweep results saved → sim_sweep.csv")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIRT alignment simulation")
    parser.add_argument("--x",      type=float, default=80,   help="Initial X offset (px)")
    parser.add_argument("--y",      type=float, default=-60,  help="Initial Y offset (px)")
    parser.add_argument("--rot",    type=float, default=15,   help="Initial rotation (deg)")
    parser.add_argument("--noise",  type=float, default=1.5,  help="Noise level (px)")
    parser.add_argument("--iters",  type=int,   default=100,  help="Max iterations")
    parser.add_argument("--sweep",  action="store_true",      help="Run parameter sweep")
    parser.add_argument("--no-gui", action="store_true",      help="Headless mode")
    parser.add_argument("--speed",  type=int,   default=50,   help="Frame delay ms (lower=faster)")
    args = parser.parse_args()

    if args.sweep:
        run_parameter_sweep()
    else:
        run_simulation(
            init_x=args.x,
            init_y=args.y,
            init_rot=args.rot,
            noise_px=args.noise,
            max_iter=args.iters,
            show_video=not args.no_gui,
            save_csv=True,
            delay_ms=args.speed,
        )
