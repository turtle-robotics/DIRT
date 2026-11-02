# DIRT

DIRT is a rover platform for autonomous soil sampling. The repository contains
the rover control modules, navigation and sensing simulations, a Panda3D
visualizer, an on-rover touchscreen display, CAD references, and
machine-learning model helpers.

## Features

- Autonomous waypoint navigation and mission sequencing
- Drill and spectrometer simulation
- GPS/navigation, distance-sensor, and vision-alignment modules
- Headless rover physics simulation
- Interactive Panda3D 3D rover viewer
- Touchscreen field display for the Pi 5 (soil data, temperatures, LiDAR,
  e-stop, sample logging)
- Raspberry Pi hardware integration points
- Optional local language-model loading helpers

## Requirements

- Python 3.10 or newer
- NumPy
- OpenCV
- gpiozero
- scikit-learn
- smbus2
- pyserial
- Panda3D for the 3D viewer

Raspberry Pi GPIO support is installed only on Linux. ROS 2 support is
optional.

## Setup

Create and activate a virtual environment, then install the project
dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or Raspberry Pi, use the equivalent activation command:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running the project

### Main mission entry point

Run the complete simulated mission:

```bash
python main.py
```

Run a subsystem demonstration:

```bash
python main.py --test alignment
python main.py --test spectrometer
python main.py --test distance
python main.py --test navigation
```

The real-hardware path is controlled by `config.py`. Review the configuration
and platform settings before connecting hardware.

### Headless rover simulation

Run one simulation:

```bash
python simulation/dirt_sim_core.py
```

Select terrain and control profiles:

```bash
python simulation/dirt_sim_core.py \
  --terrain-mode rocky \
  --control-mode aggressive \
  --csv runs/rocky.csv
```

Run all terrain and control combinations:

```bash
python simulation/dirt_sim_core.py --batch --out sim_runs/
```

Generate deterministic random waypoints:

```bash
python simulation/dirt_sim_core.py --random-waypoints 6 --seed 3
```

Available terrain profiles are `farmland`, `grassland`, `rolling`, `dunes`,
`rocky`, and `testyard`. Available control profiles are `smooth`, `pursuit`,
and `aggressive`.

### Panda3D 3D viewer

Launch the interactive viewer:

```bash
python simulation/rover_3d_panda.py
```

Useful options:

```bash
python simulation/rover_3d_panda.py --terrain-mode testyard
python simulation/rover_3d_panda.py --camera overhead
python simulation/rover_3d_panda.py --headless --iters 600
python simulation/rover_3d_panda.py --snapshot rover.png
```

Viewer controls include:

- Click: add a waypoint
- `W` / `S`: throttle
- `A` / `D`: steering
- `M`: manual mode
- `1`: autonomous waypoint mode
- `2`: mission mode
- `C`: change camera
- `Space`: pause
- `R`: reset
- `Q`: quit

### Touchscreen field display

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

## Repository layout

| Directory or file | Purpose |
| --- | --- |
| `core/` | Distance sensing and alignment-related core modules |
| `navigation/` | GPS waypoint and navigation logic |
| `simulation/` | Headless physics, alignment, spectrometer, and 3D simulations |
| `spectrometer/` | Spectrometer interfaces and simulation support |
| `lm_models/` | Optional language-model loading helpers |
| `DIRT Display/` | Touchscreen field display, ROS 2 helper node, and Pi setup script |
| `CAD/` | Mechanical CAD files and assemblies |
| `Sim_Data/` | Simulation datasets and generated analysis data |
| `tests/` | Automated tests |
| `config.py` | Platform and hardware configuration |
| `main.py` | Main mission and subsystem-test entry point |
| `requirements.txt` | Python dependency declarations |

## Testing and validation

Run the test suite from the repository root:

```bash
python -m pytest
```

Compile-check all Python modules without running hardware or GUI code:

```bash
python -m compileall .
```

Some tests and demonstrations require optional hardware, camera, display, or
Panda3D functionality. Use the headless simulation for environments without a
display.

## Data and generated files

The display's helper node writes samples to `~/dirt_samples/` on the Pi, not
into the repository.

Simulation runs write CSV logs containing pose, terrain, motor, power, and
mission data. Prefer writing new runs into a separate output directory such as
`sim_runs/` rather than overwriting checked-in reference data.

Do not commit virtual environments, Python bytecode, API keys, or hardware
credentials.
