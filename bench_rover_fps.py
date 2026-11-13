"""Frame-rate benchmark for the OpenCV rover sim (``simulation/rover_3d_sim.py``).

Steps and renders the sim at 1280x720 for 3 seconds and prints frames, time
and average FPS. Run from the repository root::

    python bench_rover_fps.py
"""

import time
from simulation.rover_3d_sim import Rover3DSim

sim = Rover3DSim()
sim.render((1280, 720))

start = time.perf_counter()
frames = 0
end_time = start + 3.0

while time.perf_counter() < end_time:
    sim.step(0.016)
    sim.render((1280, 720))
    frames += 1

elapsed = time.perf_counter() - start
print(f"frames={frames}")
print(f"elapsed={elapsed:.3f}s")
print(f"fps={frames / elapsed:.2f}")
