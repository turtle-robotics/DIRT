"""
DIRT Bot - Spectrometer Simulation (matplotlib backend)
========================================================

Mirrors the structure of simulation/rover_3d_panda.py so the two sims read
the same way: dataclasses for state/samples, a *Log class that records each
step and writes a CSV, `logging` instead of print, and an argparse CLI with
the same flag naming (--label, --csv, --iters).

This does not require the rest of the DIRT project (no `config` module, no
package imports) - it's self-contained so it runs standalone in VS Code. It
reuses the same soil base spectra, noise model, preprocessing
(Savitzky-Golay + SNV), and PLS-with-numpy-fallback regression as
spectrometer/spectrometer.py, so the numbers you see are representative of
the real pipeline, not a toy.

Run directly:
    python spectrometer_sim.py
    python spectrometer_sim.py --soil clay --interval 1.5
    python spectrometer_sim.py --soil sand --iters 40 --csv sim_spectrometer_log.csv
    python spectrometer_sim.py --snapshot spectrometer_snapshot.png
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

log = logging.getLogger("spectrometer-sim")


WAVELENGTHS_NM = (
    410, 435, 460, 485, 510, 535,
    560, 585, 610, 645, 680, 705,
    730, 760, 810, 860, 900, 940,
)

SOIL_BASES = {
    "clay": np.array([0.05, 0.06, 0.07, 0.10, 0.15, 0.22,
                       0.30, 0.38, 0.42, 0.48, 0.52, 0.55,
                       0.58, 0.62, 0.65, 0.68, 0.70, 0.72]),
    "sand": np.array([0.12, 0.18, 0.25, 0.33, 0.40, 0.46,
                       0.51, 0.55, 0.59, 0.63, 0.66, 0.68,
                       0.70, 0.72, 0.74, 0.75, 0.76, 0.77]),
    "loam": np.array([0.08, 0.10, 0.13, 0.17, 0.23, 0.29,
                       0.35, 0.41, 0.45, 0.50, 0.53, 0.56,
                       0.60, 0.63, 0.66, 0.69, 0.71, 0.73]),
}

SOIL_TYPES = tuple(sorted(SOIL_BASES.keys()))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ------------------ State / sample records (mirrors RoverState/RoverSample) ------------------


@dataclass
class SpectralReading:
    channels: Dict[int, float]
    timestamp: float = field(default_factory=time.time)
    temperature_c: float = 25.0

    def to_array(self) -> np.ndarray:
        return np.array([self.channels[wl] for wl in sorted(self.channels)])


@dataclass
class SoilAnalysisResult:
    moisture_pct: float
    organic_matter_pct: float
    iron_oxide_idx: float
    confidence: float
    raw_spectrum: SpectralReading


@dataclass
class SpectrometerSample:
    step: int
    run_label: str
    soil_type: str
    soil_moisture_pct: float
    organic_matter_pct: float
    iron_oxide_idx: float
    spectrometer_confidence: float
    spectrometer_temp_c: float


@dataclass
class SpectrometerLog:
    rows: List[SpectrometerSample] = field(default_factory=list)

    def record(
        self,
        step: int,
        run_label: str,
        soil_type: str,
        soil_result: SoilAnalysisResult,
    ) -> None:
        self.rows.append(
            SpectrometerSample(
                step=step,
                run_label=run_label,
                soil_type=soil_type,
                soil_moisture_pct=round(soil_result.moisture_pct, 1),
                organic_matter_pct=round(soil_result.organic_matter_pct, 1),
                iron_oxide_idx=round(soil_result.iron_oxide_idx, 3),
                spectrometer_confidence=round(soil_result.confidence, 3),
                spectrometer_temp_c=round(soil_result.raw_spectrum.temperature_c, 2),
            )
        )

    def save_csv(self, path: str = "sim_spectrometer_log.csv") -> None:
        if not self.rows:
            return
        with open(path, "w", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=list(self.rows[0].__dict__.keys()))
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row.__dict__)
        log.info("Spectrometer log saved -> %s", path)


# ------------------ Driver / preprocessing / model (same pipeline as spectrometer.py) ------------------


class MockAS7265xDriver:
    def __init__(self, soil_type: str = "loam", scale: float = 10_000.0,
                 rng: Optional[np.random.Generator] = None):
        self._base = SOIL_BASES.get(soil_type, SOIL_BASES["loam"])
        self._scale = scale
        self._rng = rng or np.random.default_rng()

    def read_calibrated(self) -> Dict[int, float]:
        noise = self._rng.normal(0, 0.01, len(WAVELENGTHS_NM))
        counts = np.clip((self._base + noise) * self._scale, 0, None)
        return {wl: float(c) for wl, c in zip(WAVELENGTHS_NM, counts)}

    def read_temperature(self) -> float:
        return 25.0 + float(self._rng.normal(0, 0.5))


def snv(spectrum: np.ndarray) -> np.ndarray:
    mu, std = spectrum.mean(), spectrum.std()
    return (spectrum - mu) / (std + 1e-9)


def savitzky_golay(spectrum: np.ndarray, window: int = 5, poly: int = 2) -> np.ndarray:
    half = window // 2
    out = spectrum.copy()
    for i in range(half, len(spectrum) - half):
        segment = spectrum[i - half: i + half + 1]
        x = np.arange(-half, half + 1)
        coeffs = np.polyfit(x, segment, poly)
        out[i] = np.polyval(coeffs, 0)
    return out


def preprocess(spectrum: np.ndarray) -> np.ndarray:
    return snv(savitzky_golay(spectrum))


class SoilPLSModel:
    def __init__(self, n_components: int = 6):
        try:
            from sklearn.cross_decomposition import PLSRegression
            from sklearn.preprocessing import StandardScaler
            self._pls = PLSRegression(n_components=n_components)
            self._scaler = StandardScaler()
        except Exception:
            self._pls = None
            self._scaler = None
        self._trained = False
        self._X_train = None
        self._coeffs = None
        self._X_mean = None
        self._X_std = None
        self._cov_inv = None

    def fit(self, X_raw: np.ndarray, y: np.ndarray):
        X_pre = np.apply_along_axis(preprocess, 1, X_raw)
        if self._pls is not None:
            X_sc = self._scaler.fit_transform(X_pre)
            self._pls.fit(X_sc, y)
            self._X_train = X_sc
        else:
            X_mean = X_pre.mean(axis=0)
            X_std = X_pre.std(axis=0) + 1e-9
            X_sc = (X_pre - X_mean) / X_std
            X_aug = np.hstack([np.ones((X_sc.shape[0], 1)), X_sc])
            coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            self._coeffs, self._X_mean, self._X_std, self._X_train = coeffs, X_mean, X_std, X_sc
        self._trained = True
        cov = np.cov(self._X_train.T) + np.eye(self._X_train.shape[1]) * 1e-6
        self._cov_inv = np.linalg.inv(cov)
        log.info("Soil PLS model fitted (%s backend)", "sklearn" if self._pls is not None else "numpy fallback")

    def predict(self, spectrum: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self._trained:
            raise RuntimeError("Model not trained")
        x_pre = preprocess(spectrum)
        if self._pls is not None:
            x_sc = self._scaler.transform(x_pre.reshape(1, -1))
            y_hat = self._pls.predict(x_sc)[0]
        else:
            x_sc = (x_pre - self._X_mean) / (self._X_std + 1e-9)
            x_aug = np.concatenate([[1.0], x_sc])
            y_hat = x_aug @ self._coeffs
            x_sc = x_sc.reshape(1, -1)
        mean = self._X_train.mean(axis=0)
        diff = x_sc[0] - mean
        maha2 = float(diff @ self._cov_inv @ diff)
        conf = round(min(1.0, max(0.0, float(np.exp(-0.05 * maha2)))), 3)
        return y_hat, conf

    @staticmethod
    def make_synthetic_calibration(n: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(0)
        soils = list(SOIL_TYPES)
        X, y = [], []
        for i in range(n):
            base = SOIL_BASES[soils[i % len(soils)]]
            spectrum = (base + rng.normal(0, 0.02, len(base))) * 10000.0
            X.append(spectrum)
            y.append([10 + 30 * rng.rand(), 1 + 5 * rng.rand(), 0.1 + 0.4 * rng.rand()])
        return np.vstack(X), np.vstack(y)


# ------------------ Sim app (mirrors Rover3DPandaApp's lifecycle/controls) ------------------


class SpectrometerSimApp:
    """Live matplotlib viewer over the mock AS7265x driver + PLS model.

    Structured to match Rover3DPandaApp: a bound keyboard control scheme,
    a step counter, a *Log that accumulates samples and writes CSV on
    exit, and a run_label carried through every record for later joins
    against the rover's own CSV output.
    """

    def __init__(
        self,
        soil_type: str = "loam",
        interval_s: float = 1.0,
        run_label: str = "demo",
        max_steps: Optional[int] = None,
        snapshot_path: Optional[str] = None,
    ):
        if soil_type not in SOIL_BASES:
            soil_type = "loam"

        self.soil_type = soil_type
        self.interval_s = interval_s
        self.run_label = run_label
        self.max_steps = max_steps
        self.snapshot_path = snapshot_path

        self.driver = MockAS7265xDriver(soil_type=soil_type)
        self.model = SoilPLSModel()
        X, y = SoilPLSModel.make_synthetic_calibration()
        self.model.fit(X, y)

        self.step_idx = 0
        self.paused = False
        self.finished = False
        self._snapshot_taken = False
        self.log = SpectrometerLog()

        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        self.fig.canvas.manager.set_window_title("DIRT spectrometer - live sim")
        self.bars = self.ax.bar(range(len(WAVELENGTHS_NM)), [0] * len(WAVELENGTHS_NM), color="#2a78d6")
        self.ax.set_xticks(range(len(WAVELENGTHS_NM)))
        self.ax.set_xticklabels([f"{w}" for w in WAVELENGTHS_NM], rotation=45, fontsize=8)
        self.ax.set_xlabel("wavelength (nm)")
        self.ax.set_ylabel("reflectance (counts)")
        self.ax.set_ylim(0, 12000)
        self.title = self.ax.set_title("")
        self.help_text = self.fig.text(
            0.01, 0.01,
            "1/2/3 soil (loam/sand/clay)   space pause   r reset   s snapshot   q quit",
            fontsize=8,
        )

        self._bind_controls()

    # -- controls (mirrors Rover3DPandaApp._bind_controls) --

    def _bind_controls(self) -> None:
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event) -> None:
        if event.key == "q":
            self._request_quit()
        elif event.key == " ":
            self.toggle_pause()
        elif event.key == "r":
            self.reset_pose()
        elif event.key == "s":
            self.capture_snapshot()
        elif event.key in ("1", "2", "3"):
            self.set_soil_type(SOIL_TYPES[int(event.key) - 1])

    def _request_quit(self) -> None:
        self.finished = True
        plt.close(self.fig)

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def reset_pose(self) -> None:
        self.step_idx = 0
        self.log = SpectrometerLog()
        self.driver = MockAS7265xDriver(soil_type=self.soil_type)
        log.info("Sim reset (soil=%s)", self.soil_type)

    def set_soil_type(self, soil_type: str) -> None:
        self.soil_type = soil_type
        self.driver = MockAS7265xDriver(soil_type=soil_type)
        log.info("Soil type changed -> %s", soil_type)

    def capture_snapshot(self, path: Optional[str] = None) -> bool:
        output_path = path or self.snapshot_path or "spectrometer_snapshot.png"
        try:
            self.fig.savefig(output_path, dpi=150)
        except Exception:
            log.exception("Failed to save snapshot -> %s", output_path)
            return False
        log.info("Snapshot saved -> %s", output_path)
        self._snapshot_taken = True
        return True

    # -- step loop (mirrors Rover3DPandaApp._update_task) --

    def _step(self) -> SoilAnalysisResult:
        reading = SpectralReading(
            channels=self.driver.read_calibrated(),
            temperature_c=self.driver.read_temperature(),
        )
        arr = reading.to_array()
        preds, conf = self.model.predict(arr)
        moisture = float(_clamp(preds[0], 0, 100))
        organic = float(_clamp(preds[1], 0, 100))
        iron = float(_clamp(preds[2], 0, 1))
        return SoilAnalysisResult(
            moisture_pct=moisture,
            organic_matter_pct=organic,
            iron_oxide_idx=iron,
            confidence=conf,
            raw_spectrum=reading,
        )

    def _update(self, _frame):
        if self.finished:
            return list(self.bars) + [self.title]

        if self.paused:
            return list(self.bars) + [self.title]

        result = self._step()
        arr = result.raw_spectrum.to_array()

        for bar, val in zip(self.bars, arr):
            bar.set_height(val)
        self.title.set_text(
            f"soil={self.soil_type}  temp={result.raw_spectrum.temperature_c:.1f}C  |  "
            f"moisture={result.moisture_pct:.1f}%  organic={result.organic_matter_pct:.1f}%  "
            f"iron_idx={result.iron_oxide_idx:.3f}  conf={result.confidence:.2f}"
        )

        self.log.record(self.step_idx, self.run_label, self.soil_type, result)
        self.step_idx += 1

        if self.snapshot_path and not self._snapshot_taken:
            self.capture_snapshot(self.snapshot_path)

        if self.max_steps is not None and self.step_idx >= self.max_steps:
            self.finished = True
            plt.close(self.fig)

        return list(self.bars) + [self.title]

    def run(self) -> None:
        self.ani = FuncAnimation(
            self.fig, self._update, interval=self.interval_s * 1000,
            blit=False, cache_frame_data=False,
        )
        plt.tight_layout()
        plt.show()

    def save_readings_csv(self, path: str = "sim_spectrometer_log.csv") -> None:
        self.log.save_csv(path)


def run_spectrometer_sim_demo(
    soil_type: str = "loam",
    interval_s: float = 1.0,
    max_steps: Optional[int] = None,
    run_label: str = "demo",
    csv_path: str = "sim_spectrometer_log.csv",
    save_csv: bool = True,
    snapshot_path: Optional[str] = None,
) -> SpectrometerLog:
    app = SpectrometerSimApp(
        soil_type=soil_type,
        interval_s=interval_s,
        run_label=run_label,
        max_steps=max_steps,
        snapshot_path=snapshot_path,
    )
    log.info("Starting spectrometer sim (soil=%s, interval=%.2fs)", soil_type, interval_s)
    if max_steps is None:
        log.info("Simulation end condition: window closed or 'q' pressed")
    else:
        log.info("Simulation end condition: max_steps=%d, window closed, or 'q' pressed", max_steps)

    try:
        app.run()
    finally:
        if save_csv:
            app.save_readings_csv(csv_path)

    if app.log.rows:
        last = app.log.rows[-1]
        print("\nSpectrometer Simulation Summary")
        print(f"  Steps             : {len(app.log.rows)}")
        print(f"  Soil type         : {last.soil_type}")
        print(f"  Final moisture    : {last.soil_moisture_pct:.1f}%")
        print(f"  Final organic     : {last.organic_matter_pct:.1f}%")
        print(f"  Final iron idx    : {last.iron_oxide_idx:.3f}")
        print(f"  Final confidence  : {last.spectrometer_confidence:.2f}")
        print()

    return app.log


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

    parser = argparse.ArgumentParser(description="DIRT spectrometer simulation (matplotlib)")
    parser.add_argument("--soil", choices=list(SOIL_TYPES), default="loam", help="Soil type to simulate")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between captures")
    parser.add_argument("--iters", type=int, default=0, help="Max simulation steps; 0 means no fixed step limit")
    parser.add_argument("--csv", type=str, default="sim_spectrometer_log.csv", help="CSV output path")
    parser.add_argument("--label", type=str, default="demo", help="Run label written into the CSV")
    parser.add_argument("--snapshot", type=str, default=None, help="Save a PNG snapshot to this path on first frame")
    args = parser.parse_args()

    log.info("Controls: 1/2/3 soil type, space pause, r reset, s snapshot, q quit")
    run_spectrometer_sim_demo(
        soil_type=args.soil,
        interval_s=args.interval,
        max_steps=None if args.iters <= 0 else args.iters,
        run_label=args.label,
        csv_path=args.csv,
        save_csv=True,
        snapshot_path=args.snapshot,
    )