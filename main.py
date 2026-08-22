"""
DIRT Bot – Main Entry Point
=============================
Runs the complete autonomous soil sampling sequence:
  GPS navigate → SLAM fine-position → Vision align → Drill → Measure → Repeat

Usage:
  # Full simulation (no hardware):
  python main.py

  # Real hardware:
  # 1. Edit config.py → set PLATFORM = Platform.RASPBERRY_PI
  # 2. python main.py --waypoints waypoints.csv

  # Single-subsystem tests:
  python main.py --test alignment
  python main.py --test spectrometer
  python main.py --test distance
  python main.py --test navigation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-24s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("DIRT")


def _run_alignment_test():
    """Quick alignment test – works in simulation with no hardware."""
    from simulation.alignment_sim import run_simulation
    log.info("Running alignment simulation test…")
    clog = run_simulation(
        init_x=80, init_y=-60, init_rot=15,
        noise_px=1.5, max_iter=100,
        show_video=True, save_csv=True,
    )
    clog.print_summary()


def _run_spectrometer_test():
    """Spectrometer end-to-end test (mock or real hardware)."""
    from spectrometer.spectrometer import Spectrometer
    log.info("Running spectrometer test…")
    with Spectrometer() as spec:
        log.info("Capturing white reference (simulated)…")
        spec.calibrate_white(n=3)

        log.info("Taking 3 measurements…")
        for i in range(3):
            result = spec.measure()
            print(f"\nMeasurement {i+1}:")
            print(result.report())
            # Show raw spectrum
            arr = result.raw_spectrum.to_array()
            print(f"  Spectrum: min={arr.min():.0f}  max={arr.max():.0f}  "
                  f"mean={arr.mean():.0f}")


def _run_distance_test():
    """Distance sensor session capture."""
    from core.distance_sensor import DistanceSensor
    log.info("Running distance sensor test…")
    with DistanceSensor() as sensor:
        session = sensor.capture_session(duration_s=5.0, sample_hz=5.0)
        print("\nSession summary:")
        for k, v in session.summary().items():
            print(f"  {k}: {v}")
        # Save CSV
        Path("distance_session.csv").write_text(session.to_csv())
        log.info("Saved → distance_session.csv")


def _run_navigation_test():
    """Full navigation stack demo with synthetic waypoints."""
    from navigation.navigator import Navigator, Waypoint, GPSCoord
    log.info("Running navigation demo…")
    waypoints = [
        Waypoint("Alpha",   GPSCoord(30.6281, -96.3344)),
        Waypoint("Bravo",   GPSCoord(30.6283, -96.3341)),
        Waypoint("Charlie", GPSCoord(30.6285, -96.3338)),
    ]
    nav = Navigator(waypoints)
    nav.run_demo(ticks=120)


def _run_mission_from_csv(path: str):
    """Run a waypoint mission from a CSV list of field targets."""
    from navigation.navigator import Navigator

    if not path:
        raise ValueError("A mission CSV path is required")

    nav = Navigator.from_csv(path)
    log.info("=== Mission start: %d waypoints from %s ===", len(nav.waypoints), path)
    results = nav.run_mission(ticks=120, tick_delay=0.2)
    if results:
        print("\nMission results:")
        for item in results:
            print(f"  - {item['waypoint']}: {item['status']} ({item['distance_m']} m)")
        log.info("Mission completed with %d waypoint(s)", len(results))
    else:
        log.warning("No waypoints reached during this mission")


def _run_rover_3d_test():
    """Full 3D rover motion demo over terrain."""
    from simulation.rover_3d_sim import run_rover_3d_demo
    log.info("Running 3D rover demo…")
    run_rover_3d_demo(show_video=True, save_csv=True)


def _load_waypoints_csv(path: str):
    """Load waypoints from a CSV with columns: name,lat,lon"""
    from navigation.navigator import Waypoint, GPSCoord
    waypoints = []
    with open(path) as f:
        for row in csv.DictReader(f):
            waypoints.append(Waypoint(
                name=row["name"],
                gps=GPSCoord(float(row["lat"]), float(row["lon"])),
            ))
    log.info("Loaded %d waypoints from %s", len(waypoints), path)
    return waypoints


def _run_full_mission(waypoints_csv: Optional[str] = None):
    """
    Full autonomous mission:
    GPS navigate → SLAM → Vision align → Drill → Measure → next waypoint
    """
    from navigation.navigator import Navigator, Waypoint, GPSCoord, NavState
    from core.alignment_system import AlignmentControlSystem, SystemState
    from spectrometer.spectrometer import Spectrometer
    from core.distance_sensor import DistanceSensor

    if waypoints_csv:
        waypoints = _load_waypoints_csv(waypoints_csv)
    else:
        waypoints = [
            Waypoint("Alpha",   GPSCoord(30.6281, -96.3344)),
            Waypoint("Bravo",   GPSCoord(30.6283, -96.3341)),
        ]

    nav  = Navigator(waypoints)
    spec = Spectrometer()
    spec.calibrate_white(n=3)

    results = []
    nav.state = nav.state.__class__.IDLE   # reset

    log.info("=== DIRT Bot Mission Start: %d waypoints ===", len(waypoints))

    try:
        while nav.current_waypoint is not None:
            state = nav.step()

            if state == NavState.ALIGNING:
                # Start vision alignment system
                with AlignmentControlSystem(show_video=True) as acs:
                    acs.start()
                    while acs.state not in (SystemState.ALIGNED, SystemState.ERROR):
                        time.sleep(0.1)
                    if acs.state == SystemState.ALIGNED:
                        nav.notify_aligned()
                    else:
                        log.error("Alignment failed – skipping waypoint")
                        nav.current_wp_idx += 1
                        nav.state = NavState.GPS_NAVIGATE

            elif state == NavState.DRILLING:
                log.info("Drill sequence started…")
                # Distance sensor monitors drill depth
                with DistanceSensor() as ds:
                    target_depth_cm = 15.0
                    while True:
                        d = ds.read_cm()
                        log.info("Drill depth: %.1f cm", d)
                        if d <= target_depth_cm:
                            break
                        time.sleep(0.5)
                nav.notify_drill_complete()

            elif state == NavState.MEASURING:
                result = spec.measure()
                wp     = nav.current_waypoint
                log.info("Measured at %s:\n%s", wp.name if wp else "?", result.report())
                results.append({
                    "waypoint":   wp.name if wp else "?",
                    "moisture":   result.moisture_pct,
                    "org_matter": result.organic_matter_pct,
                    "iron_oxide": result.iron_oxide_idx,
                    "confidence": result.confidence,
                })
                nav.notify_measure_complete()

            elif state == NavState.DONE:
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        log.info("Mission interrupted by user")

    # Save results
    if results:
        out = Path("mission_results.json")
        out.write_text(json.dumps(results, indent=2))
        log.info("Mission results saved → %s", out)
        print("\n=== Mission Results ===")
        for r in results:
            print(f"  {r['waypoint']}: moisture={r['moisture']:.1f}%  "
                  f"OM={r['org_matter']:.1f}%  Fe={r['iron_oxide']:.3f}  "
                  f"conf={r['confidence']:.2f}")

    log.info("=== Mission complete ===")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from typing import Optional

    parser = argparse.ArgumentParser(description="DIRT Bot")
    parser.add_argument(
        "--test",
        choices=["alignment", "spectrometer", "distance", "navigation", "rover3d"],
        help="Run a single subsystem test",
    )
    parser.add_argument(
        "--waypoints",
        type=str,
        default=None,
        help="Path to waypoints CSV (name,lat,lon)",
    )
    parser.add_argument(
        "--mission",
        type=str,
        default=None,
        help="Run a farm mission from a waypoint CSV with columns: name,lat,lon,action,tolerance_m",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run alignment simulation parameter sweep",
    )
    args = parser.parse_args()

    if args.test == "alignment":
        _run_alignment_test()
    elif args.test == "spectrometer":
        _run_spectrometer_test()
    elif args.test == "distance":
        _run_distance_test()
    elif args.test == "navigation":
        _run_navigation_test()
    elif args.test == "rover3d":
        _run_rover_3d_test()
    elif args.sweep:
        from simulation.alignment_sim import run_parameter_sweep
        run_parameter_sweep()
    elif args.mission:
        _run_mission_from_csv(args.mission)
    else:
        _run_full_mission(args.waypoints)
