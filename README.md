# DIRT

DIRT is a rover platform for autonomous soil sampling. The repository contains
the rover control modules, navigation and sensing simulations, a Panda3D
visualizer, CAD references, and machine-learning model helpers.

## Features

- Autonomous waypoint navigation and mission sequencing
- Drill and spectrometer simulation
- GPS/navigation, distance-sensor, and vision-alignment modules
- Headless rover physics simulation
- Interactive Panda3D 3D rover viewer
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

## Repository layout

| Directory or file | Purpose |
| --- | --- |
| `core/` | Distance sensing and alignment-related core modules |
| `navigation/` | GPS waypoint and navigation logic |
| `simulation/` | Headless physics, alignment, spectrometer, and 3D simulations |
| `spectrometer/` | Spectrometer interfaces and simulation support |
| `lm_models/` | Optional language-model loading helpers |
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

Simulation runs write CSV logs containing pose, terrain, motor, power, and
mission data. Prefer writing new runs into a separate output directory such as
`sim_runs/` rather than overwriting checked-in reference data.

Do not commit virtual environments, Python bytecode, API keys, or hardware
credentials.
