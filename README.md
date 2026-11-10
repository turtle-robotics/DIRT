# DIRT: Diagnostic Inspection Robot for Terrain

DIRT is an autonomous soil-diagnostic rover built by the DIRT project team at
TURTLE Robotics (Texas A&M University). It drives
to waypoints across a field, drills a subsurface soil sample, and runs an
on-the-spot spectral analysis, so soil quality can be checked faster, more
densely, and more cheaply than hand sampling and lab testing.

This repository holds the rover software: mission logic, navigation, the
spectrometer and PLS soil model, the Panda3D rover simulation, the Pi 5
touchscreen display, and reference CAD.

**Status (Fall 2026):** Preliminary Design Review held Nov 8, 2026.
Spring 2027 is the prototyping phase: printed hull and treads, drive test, and
drill test.

## Contents

- [How the rover works](#how-the-rover-works)
- [Hardware](#hardware)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Running the project](#running-the-project)
- [Testing](#testing)
- [Known issues and gaps](#known-issues-and-gaps)
- [Contributing](#contributing)
- [Team](#team)

## How the rover works

Each mission loops through the same steps:

1. **Load mission:** waypoints for the field area.
2. **Navigate:** drive to the next waypoint and stop within 5 m.
3. **Drill:** lower the auger to a sampling depth of 10 to 13 cm.
4. **Scoop:** a ceiling-mounted pantograph scoop lowers and grabs the drilled soil.
5. **Measure:** read the sample with the spectrometer in a light-isolated chamber.
6. **Eject:** dump the soil through a trapdoor so the next reading is clean.
7. **Log:** save spectra, GPS position, and time.
8. **Repeat** at the next waypoint, or return home.

The emergency stop is live at every step (see [Hardware](#hardware)).

### Software architecture (planned ROS 2 stack)

The rover runs **ROS 2 Jazzy on a Raspberry Pi 5** and **micro-ROS on an
ESP32**, linked over USB serial.

```
Raspberry Pi 5 (ROS 2 Jazzy)
  LD19 driver ──/scan──▶ Nav2 (costmap + controller) ──/cmd_vel──▶ micro-ROS agent
  IMU node ────/imu───▶ EKF (robot_localization) ──/odom──▶ Nav2
  micro-ROS agent ──/wheel_odom──▶ EKF
                         │ USB serial
ESP32 (micro-ROS)        ▼
  micro-ROS client (cmd_vel in, encoder ticks out)
    ──▶ track mixer (v, ω to left/right speed) ──▶ motor I/O (PWM + encoders)
```

The touchscreen display connects to this stack through rosbridge (see
[Touchscreen field display](#touchscreen-field-display-dirt-display)).

> The Nav2, EKF, and micro-ROS packages and the ESP32 firmware are not in
> this repository yet. See [Known issues and gaps](#known-issues-and-gaps).

## Hardware

| Subsystem | Part | Notes |
| --- | --- | --- |
| Compute | Raspberry Pi 5 | ROS 2 Jazzy, Nav2, EKF; hosts LiDAR, IMU, GPS, spectrometer, display |
| Motor controller | ESP32-DevKitC | micro-ROS; motor PWM/DIR, encoders, e-stop relay |
| Motor drivers | 2× Cytron MDD10A | One for the two tracks, one for the drill motors |
| Drive | 2× REV HD Hex Motor + 40:1 spur gearbox | 150 RPM, 4.2 N·m stall; one per tank tread |
| Battery | Renogy RBT1220LFP, 12.8 V 20 Ah LiFePO4 | 256 Wh, about 2.2 h at the 83 W high load; 20 A continuous limit |
| LiDAR | LD19 | Obstacle scans for the Nav2 costmap |
| IMU | ICM-20948 (breakout) | Heading and rates for the EKF, on Pi I2C |
| GPS | NEO-M8N | Position tag for every sample, on Pi UART |
| Spectrometer | StellarNet EPP2000-HR (NIR2) | 900 to 1075 nm, 2048-pixel CCD, USB 2.0 to the Pi 5 |

**Power and safety.** Battery → ~20 A main fuse → power switch → e-stop relay →
MDD10A drivers → motors. The 5 V logic rail (buck to the Pi 5, ESP32, and
LiDAR) splits off after the power switch, so the computers stay on during a
stop. Three things can open the relay: the red mushroom button on the hull, a
remote STOP from the display or a phone (`/dirt/estop`), or a detected fault
(drill stall, low battery, over-current, tip-over, lost comms). The stop
latches until someone releases the button and does a manual reset. A software
interlock blocks drilling while the tracks move, which keeps peak current under
the battery's 20 A limit.

Mechanical CAD is designed in SolidWorks. Reference parts are in [`CAD/`](CAD/).

## Repository layout

| Path | What it is |
| --- | --- |
| `main.py` | Mission and subsystem-test entry point |
| `config.py` | All tunable parameters; `PLATFORM` switches between simulation and hardware |
| `core/` | Vision alignment control loop and distance sensor support |
| `navigation/` | GPS waypoint navigation and SLAM interface |
| `spectrometer/` | Spectrometer driver, preprocessing, and PLS soil model, with a mock backend |
| `simulation/dirt_sim_core.py` | Headless rover physics: terrain, skid-steer drive, REV motor model, mission state machine, CSV logs |
| `simulation/rover_3d_panda.py` | Panda3D 3D viewer that renders `dirt_sim_core` |
| `simulation/rover_3d_sim.py` | Older OpenCV-rendered 3D sim (replaced by the Panda3D viewer) |
| `simulation/spectrometer_sim.py` | Spectrometer simulation with a matplotlib view |
| `simulation/alignment_sim.py` | Alignment loop simulation with synthetic camera frames |
| `DIRT Display/` | Touchscreen UI for the Pi 5, its ROS 2 helper node, and the Pi setup script |
| `lm_models/` | PLS model notebooks (`PseudoMODEL.ipynb` and reference models) and an optional language-model loader |
| `CAD/` | SolidWorks parts and assemblies (`MainAssem.SLDASM`, spectrometer box and elevator, motor box, spur gears) |
| `Sim_Data/` | Reference simulation outputs |
| `tests/` | pytest tests |
| `w` | Example farm mission CSV for `main.py --mission` (columns `name,lat,lon,action,tolerance_m`) |
| `fcntl.py` | Windows-only import shim (see known issues) |
| `bench_rover_fps.py` | Frame-rate benchmark for the OpenCV sim |

## Setup

Requires **Python 3.10 or newer**. Panda3D is only needed for the 3D viewer.
`RPi.GPIO` installs only on Linux. ROS 2 (`rclpy`) comes with ROS 2 Jazzy, not
pip, and is only needed for the display helper node and the rover stack.

Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux, macOS, or Raspberry Pi:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Everything runs in simulation by default (`PLATFORM = Platform.SIMULATION` in
`config.py`), so no hardware is needed to try it.

## Running the project

### Main mission entry point

Run the complete simulated mission:

```bash
python main.py
```

Run a farm mission from a waypoint CSV:

```bash
python main.py --mission w
```

Run a single subsystem demo:

```bash
python main.py --test alignment
python main.py --test spectrometer
python main.py --test distance
python main.py --test navigation
python main.py --test rover3d
```

For real hardware, set `PLATFORM` in `config.py` and check every pin and port
there before connecting anything.

### Headless rover simulation

`simulation/dirt_sim_core.py` has no graphics dependency, so it also runs as a
batch job on the TAMU HPRC Grace cluster.

```bash
# One run
python simulation/dirt_sim_core.py

# Pick terrain and controller, write the log somewhere else
python simulation/dirt_sim_core.py --terrain-mode rocky --control-mode aggressive --csv runs/rocky.csv

# Every terrain × controller combination
python simulation/dirt_sim_core.py --batch --out sim_runs/

# Repeatable random sample sites
python simulation/dirt_sim_core.py --random-waypoints 6 --seed 3
```

Terrain profiles: `farmland`, `grassland`, `rolling`, `dunes`, `rocky`,
`testyard`. Control profiles: `smooth`, `pursuit`, `aggressive`. Other useful
flags: `--no-mission` (drive only, no drill or measure), `--mass`,
`--sprocket-radius`, `--track-gauge`, `--max-time`, `--log-every`. Run with
`--help` for the full list.

Each run logs position, heading, speed, wheel slip, power draw, path curvature,
and mission state to CSV.

### Panda3D 3D viewer

```bash
python simulation/rover_3d_panda.py
python simulation/rover_3d_panda.py --terrain-mode testyard --camera overhead
python simulation/rover_3d_panda.py --headless --iters 600
python simulation/rover_3d_panda.py --snapshot rover.png
```

Cameras: `follow`, `overhead`, `cockpit`, `side`. Use `--resolution 1920x1080`
on smaller screens (the default is 2560x1440).

| Key | Action |
| --- | --- |
| Click | Add a waypoint |
| `W` / `S` | Throttle |
| `A` / `D` | Steering |
| `M` | Manual mode |
| `1` | Autonomous waypoint mode |
| `2` | Mission mode |
| `C` | Change camera |
| `Space` | Pause |
| `R` | Reset |
| `Q` | Quit |

### Spectrometer and alignment sims

```bash
python simulation/spectrometer_sim.py --soil loam
python simulation/alignment_sim.py --no-gui
python simulation/alignment_sim.py --sweep
```

### Touchscreen field display (`DIRT Display/`)

`DIRT Display/` holds the touchscreen interface that runs on the rover's
Raspberry Pi 5. It shows soil readings, temperatures, LiDAR, heading, drive
commands, logged samples, and system health, and it has an on-screen e-stop.
It talks to the ROS 2 stack through rosbridge, so it is a single HTML file with
no build step.

| File | Purpose |
| --- | --- |
| `dirt_display.html` | The display. Open it in any browser. |
| `dirt_display_node.py` | ROS 2 helper node: saves samples to CSV, publishes Pi stats and network address, optional e-stop gate |
| `setup_dirt_display.sh` | One-time Pi setup: rosbridge, boot services, phone web server, full-screen kiosk |

**Try it on any computer.** Open `DIRT Display/dirt_display.html` in a browser
and choose **Demo mode** on the TURTLE start menu. Demo mode simulates the
rover and needs no ROS install. It uses the same terrain maps as the Panda3D
viewer (`farmland`, `grassland`, `rolling`, `dunes`, `rocky`, `testyard`); pick
one under Settings, Demo terrain. Each map has its own obstacles in the LiDAR
view, its own soil readings, and its own spectrum.

**Showcase mode** (for PDR, recruiting tables, and events) cycles through every
view with a caption for visitors and moves to the next terrain each loop. Start
it from the start menu or Settings. A touch pauses it, and it resumes after 60
seconds. Open the display with `#showcase` at the end of the URL to start it
automatically.

**Install on the rover Pi** (Ubuntu 24.04 + ROS 2 Jazzy):

```bash
cd "DIRT Display"
bash setup_dirt_display.sh
sudo reboot
```

After reboot the Pi opens the display full screen. The start menu checks for
rosbridge and switches to live data after 10 seconds.

**Open it on a phone.** The Pi also serves the display on port 8080. Any phone
on the same network can open `http://<pi-address>:8080/`, and Settings, Open on
a phone shows the address and a QR code. The phone gets the full display,
including the STOP button. Campus Wi-Fi usually blocks phone-to-device traffic,
so use a hotspot or travel router for the rover. Samples are saved in
`~/dirt_samples/` as `samples_YYYYMMDD.csv` and `spectra_YYYYMMDD.csv`.

To also turn on the software e-stop gate, run
`DIRT_ESTOP_GATE=true bash setup_dirt_display.sh` and remap Nav2's velocity
output to `/cmd_vel_nav`.

**Run the helper node by hand** (on the Pi or in WSL with ROS 2 Jazzy). It needs
ROS 2 sourced, not the project venv, because `rclpy` comes with ROS 2 and is not
a pip package:

```bash
source /opt/ros/jazzy/setup.bash
python3 "DIRT Display/dirt_display_node.py"
```

**ROS 2 topics.** The display subscribes to these by default. If a topic name
in the stack is different, change the `SENSORS`, `ROVER`, and `SYS` blocks at
the top of the script in `dirt_display.html`, and the matching constants at the
top of `dirt_display_node.py`.

| Topic | Type | Used for |
| --- | --- | --- |
| `/dirt/soil/temperature` | `sensor_msgs/Temperature` | Soil temperature |
| `/dirt/soil/moisture` | `std_msgs/Float32` | Moisture (% VWC) |
| `/dirt/soil/ph` | `std_msgs/Float32` | Soil pH |
| `/dirt/soil/ec` | `std_msgs/Float32` | Conductivity (µS/cm) |
| `/dirt/env/temperature` | `sensor_msgs/Temperature` | Air temperature |
| `/dirt/battery` | `sensor_msgs/BatteryState` | Battery voltage and pack temperature |
| `/dirt/pi/cpu_temperature` | `sensor_msgs/Temperature` | Pi 5 CPU (published by the helper node) |
| `/dirt/motors/left/temperature`, `/dirt/motors/right/temperature` | `sensor_msgs/Temperature` | Drive motor temperatures |
| `/dirt/spectrometer/reflectance` | `std_msgs/Float32MultiArray` | Spectrum, 400 to 1000 nm evenly spaced |
| `/dirt/pls/predictions` | `std_msgs/Float32MultiArray` | PLS output: organic carbon, total nitrogen, clay |
| `/imu`, `/odom`, `/scan`, `/cmd_vel`, `/wheel_odom` | standard types | Heading, position, LiDAR, drive, ESP32 link |
| `/dirt/pi/stats` | `std_msgs/String` (JSON) | Pi CPU, memory, disk, uptime |
| `/rosout` | `rcl_interfaces/Log` | Recent log in the System view |

The display publishes `/dirt/ui/sample_request` (`std_msgs/Empty`) when someone
taps Take sample, and `/dirt/estop` (`std_msgs/Bool`) from the STOP button.

**E-stop note.** The on-screen STOP only publishes `/dirt/estop`. The ESP32
firmware must subscribe to it and stop the motors, or the gate must be enabled.
It is a backup to the physical e-stop, not a replacement.

## Testing

From the repository root:

```bash
python -m pytest            # run the tests
python -m compileall .      # syntax-check every module without running hardware or GUI code
```

Some tests and demos need Panda3D, a camera, or a display. Use the headless
simulation on machines without one.

## Known issues and gaps

The PDR design has moved ahead of parts of this code. Before building on these
modules, check them against the hardware table above.

- **Spectrometer driver does not match the hardware.** `spectrometer/spectrometer.py`
  and `SpectrometerConfig` in `config.py` target an 18-band AS7265x over I2C
  (410 to 940 nm). The rover uses a StellarNet EPP2000-HR over USB (900 to
  1075 nm). The display also assumes a 400 to 1000 nm spectrum on
  `/dirt/spectrometer/reflectance`.
- **ROS 2 rover stack is not here yet.** Nav2 and EKF configs, launch files, the
  LD19 and IMU nodes, and the ESP32 micro-ROS firmware still need to be added.
- **Legacy hardware settings.** `config.py` still has camera, red-marker
  alignment, HC-SR04, and direct Pi GPIO motor pins from the earlier concept.
  The PDR design drives motors from the ESP32 through MDD10A drivers.
- **`fcntl.py` at the repo root shadows Python's built-in `fcntl` on Linux.**
  It is a Windows shim, but on the Pi it will break anything that needs the real
  module (for example `pyserial`). Move it or only load it on Windows before
  running on hardware.
- **Repo cleanup.** `__pycache__/` folders are committed, root-level
  `sim_*.csv` files duplicate `Sim_Data/`, and the mission file `w` has no
  extension. A `.gitignore` at the root would keep generated files out.
- **Display safety.** The on-screen STOP only publishes `/dirt/estop`. It is a
  backup to the physical e-stop, not a replacement.

## Contributing

- Make changes on a branch and open a pull request into `main`.
- Start every new file with a docstring that says what it does, and comment
  any non-obvious math (drive model, track mixer, EKF settings, spectral
  preprocessing).
- Write simulation output to a folder like `sim_runs/`, not over the checked-in
  reference data.
- Never commit virtual environments, bytecode, API keys, or hardware
  credentials.
- Put design documents, test write-ups, and meeting notes in the Documentation
  folder of the DIRT shared drive.

## Team

DIRT is a TURTLE Robotics R&D project at Texas A&M University.

**Project Lead:** Kaleb Harris

**Members:** Shourya Bhoomireddy, Chase Foster, Alexander Govea, Kekoa Iida,
Nikhil Nagboth, Anna Nguyen, Kenneth Nguyen, Amu Panchangam, Isaac Ridlen,
Chaitra Shenoy, Matthew Suprato, Andres Talamantes, Wenrui Xu
