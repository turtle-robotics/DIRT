"""DIRT spectrometer module: AS7265x driver + PLS pipeline + mock backend.

This module provides:
 - `SpectralReading` / `SoilAnalysisResult` dataclasses
 - `AS7265xDriver` (I2C) with `MockAS7265xDriver` fallback
 - `SoilPLSModel` for training/prediction (sklearn)
 - `Spectrometer` facade that ties driver -> preprocessing -> model

The implementation focuses on being runnable in SIMULATION mode without hardware.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import CFG, Platform

log = logging.getLogger(__name__)


# Wavelengths (nm) for the AS7265x 18 channels
WAVELENGTHS_NM = (
    410, 435, 460, 485, 510, 535,
    560, 585, 610, 645, 680, 705,
    730, 760, 810, 860, 900, 940,
)


@dataclass
class SpectralReading:
    channels: Dict[int, float]
    timestamp: float = field(default_factory=time.time)
    temperature_c: float = 25.0

    def to_array(self) -> np.ndarray:
        return np.array([self.channels[wl] for wl in sorted(self.channels)])

    def __repr__(self) -> str:
        vals = " | ".join(f"{wl}nm:{v:.0f}" for wl, v in sorted(self.channels.items()))
        return f"SpectralReading({vals})"


# Backwards-compatible alias expected by some sims
SpectrumReading = SpectralReading


class SimulatedSpectrometer:
    """Compatibility wrapper used by older simulation scripts.

    Provides a `capture()` method returning a `SpectrumReading` to match
    earlier interface expectations.
    """

    def __init__(self, soil_type: str = "loam"):
        self._mock = MockAS7265xDriver(soil_type=soil_type)

    def capture(self) -> SpectrumReading:
        channels = self._mock.read_calibrated()
        temp = self._mock.read_temperature()
        return SpectrumReading(channels=channels, temperature_c=temp)


@dataclass
class SoilAnalysisResult:
    moisture_pct: float
    organic_matter_pct: float
    iron_oxide_idx: float
    confidence: float
    raw_spectrum: SpectralReading

    def report(self) -> str:
        return (
            f"Moisture: {self.moisture_pct:.1f}% | "
            f"Organic matter: {self.organic_matter_pct:.1f}% | "
            f"Iron idx: {self.iron_oxide_idx:.3f} | "
            f"Conf: {self.confidence:.2f}"
        )


# ------------------ Driver layer (AS7265x) ------------------

AS7265X_ADDR = 0x49
AS7265X_WRITE_REG = 0x01
AS7265X_READ_REG = 0x02
AS7265X_STATUS_REG = 0x00
AS7265X_TX_VALID = 0x02
AS7265X_RX_VALID = 0x01
AS7265X_TAKE_MEAS = 0x08


class AS7265xDriver:
    """Minimal I2C driver using smbus2. If smbus2 is missing this will raise.

    The real device uses a virtual register protocol; this implementation
    implements the read/write loops conservatively and exposes `read_calibrated()`.
    """

    def __init__(self, cfg=None, bus_id: int = 1):
        try:
            import smbus2
        except Exception as exc:  # pragma: no cover - hardware
            raise RuntimeError("smbus2 required for AS7265xDriver") from exc

        self._bus = smbus2.SMBus(bus_id)
        # cfg may be None when running on non-hardware machines; provide defaults
        if cfg is None:
            class _C: pass
            cfg = _C()
            cfg.i2c_address = AS7265X_ADDR
            cfg.gain = 1
            cfg.integration = 1
            cfg.led_current = 1
        self._addr = getattr(cfg, "i2c_address", AS7265X_ADDR)
        self._cfg = cfg
        # perform initial configuration (gain, integration, leds)
        try:
            self._configure()
        except Exception:
            log.exception("AS7265x configuration failed")

    def _wait_for_tx(self):
        for _ in range(200):
            status = self._bus.read_byte_data(self._addr, AS7265X_STATUS_REG)
            if status & AS7265X_TX_VALID:
                return
            time.sleep(0.001)
        raise TimeoutError("AS7265x TX timeout")

    def _wait_for_rx(self):
        for _ in range(200):
            status = self._bus.read_byte_data(self._addr, AS7265X_STATUS_REG)
            if status & AS7265X_RX_VALID:
                return
            time.sleep(0.001)
        raise TimeoutError("AS7265x RX timeout")

    def _write_virt(self, reg: int, val: int):
        self._wait_for_tx()
        self._bus.write_byte_data(self._addr, AS7265X_WRITE_REG, reg | 0x80)
        self._wait_for_tx()
        self._bus.write_byte_data(self._addr, AS7265X_WRITE_REG, val)

    def _read_virt(self, reg: int) -> int:
        self._wait_for_tx()
        self._bus.write_byte_data(self._addr, AS7265X_WRITE_REG, reg)
        self._wait_for_rx()
        return self._bus.read_byte_data(self._addr, AS7265X_READ_REG)

    def _configure(self):
        cfg = self._cfg
        # These register addresses follow the common usage in drivers
        try:
            self._write_virt(0x04, cfg.gain)
            self._write_virt(0x05, cfg.integration)
            self._write_virt(0x07, cfg.led_current)
        except Exception:
            log.debug("Configuration writes failed (device may be absent)")

    def _trigger(self):
        self._write_virt(0x04, AS7265X_TAKE_MEAS)
        for _ in range(500):
            status = self._bus.read_byte_data(self._addr, AS7265X_STATUS_REG)
            if status & AS7265X_RX_VALID:
                return
            time.sleep(0.001)
        raise TimeoutError("AS7265x measurement timeout")

    def read_calibrated(self) -> Dict[int, float]:
        self._trigger()
        result: Dict[int, float] = {}
        base_reg = 0x14
        # Each calibrated channel is 4 bytes (big-endian float)
        for i, wl in enumerate(WAVELENGTHS_NM):
            try:
                b0 = self._read_virt(base_reg + i * 4 + 0)
                b1 = self._read_virt(base_reg + i * 4 + 1)
                b2 = self._read_virt(base_reg + i * 4 + 2)
                b3 = self._read_virt(base_reg + i * 4 + 3)
                bs = bytes([b0, b1, b2, b3])
                val = float(np.frombuffer(bs, dtype='>f4')[0])
                result[wl] = float(val)
            except Exception:
                # On error return zero for this channel
                result[wl] = 0.0
        return result

    def read_temperature(self) -> float:
        return float(self._read_virt(0x06))


class MockAS7265xDriver:
    """Mock driver that returns plausible soil spectra with noise."""

    _SOIL_BASES = {
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

    def __init__(self, soil_type: str = "loam", scale: float = 10_000.0):
        self._base = self._SOIL_BASES.get(soil_type, self._SOIL_BASES["loam"])
        self._scale = scale
        log.info("[AS7265x] Mock driver active (soil_type=%s)", soil_type)

    def read_calibrated(self) -> Dict[int, float]:
        noise = np.random.normal(0, 0.01, len(WAVELENGTHS_NM))
        counts = (self._base + noise) * self._scale
        counts = np.clip(counts, 0, None)
        return {wl: float(c) for wl, c in zip(WAVELENGTHS_NM, counts)}

    def read_temperature(self) -> float:
        return 25.0 + np.random.normal(0, 0.5)


def _make_driver():
    if CFG.platform == Platform.SIMULATION:
        return MockAS7265xDriver()
    try:
        return AS7265xDriver()
    except Exception as exc:
        log.warning("AS7265x hardware failed (%s) - using mock", exc)
        return MockAS7265xDriver()


# ------------------ Preprocessing helpers ------------------


def snv(spectrum: np.ndarray) -> np.ndarray:
    mu = spectrum.mean()
    std = spectrum.std()
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


# ------------------ PLS model pipeline ------------------


class SoilPLSModel:
    TARGET_NAMES = ["moisture_pct", "organic_matter_pct", "iron_oxide_idx"]

    def __init__(self, n_components: int = 6):
        try:
            from sklearn.cross_decomposition import PLSRegression  # noqa: F401
            from sklearn.preprocessing import StandardScaler  # noqa: F401
        except Exception:
            log.warning("scikit-learn not available: model training/prediction disabled")
            self._pls = None
            self._scaler = None
            self.n_components = n_components
            self._trained = False
            self._X_train = None
            return

        from sklearn.cross_decomposition import PLSRegression
        from sklearn.preprocessing import StandardScaler

        self.n_components = n_components
        self._pls = PLSRegression(n_components=n_components)
        self._scaler = StandardScaler()
        self._trained = False
        self._X_train: Optional[np.ndarray] = None

    def fit(self, X_raw: np.ndarray, y: np.ndarray):
        if self._pls is None:
            raise RuntimeError("scikit-learn required to train model")
        X_pre = np.apply_along_axis(preprocess, 1, X_raw)
        X_sc = self._scaler.fit_transform(X_pre)
        self._pls.fit(X_sc, y)
        self._X_train = X_sc
        self._trained = True
        log.info("PLS model fitted: %d samples, %d components", len(X_raw), self.n_components)

    def predict(self, spectrum: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self._trained or self._pls is None:
            raise RuntimeError("Model not trained or sklearn not installed")
        x_pre = preprocess(spectrum).reshape(1, -1)
        x_sc = self._scaler.transform(x_pre)
        y_hat = self._pls.predict(x_sc)[0]
        conf = self._hotelling_confidence(x_sc)
        return y_hat, conf

    def _hotelling_confidence(self, x_sc: np.ndarray) -> float:
        if self._X_train is None:
            return 0.0
        mean = self._X_train.mean(axis=0)
        diff = x_sc[0] - mean
        cov = np.cov(self._X_train.T) + np.eye(self._X_train.shape[1]) * 1e-6
        maha2 = float(diff @ np.linalg.inv(cov) @ diff)
        conf = float(np.exp(-0.05 * maha2))
        return round(min(1.0, max(0.0, conf)), 3)

    @staticmethod
    def make_synthetic_calibration(n: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(0)
        soils = ["loam", "sand", "clay"]
        X = []
        y = []
        for i in range(n):
            st = soils[i % len(soils)]
            base = MockAS7265xDriver._SOIL_BASES[st]
            variability = np.random.normal(0, 0.02, len(base))
            spectrum = (base + variability) * 10000.0
            moisture = 10 + 30 * rng.rand()
            om = 1 + 5 * rng.rand()
            iron = 0.1 + 0.4 * rng.rand()
            X.append(spectrum)
            y.append([moisture, om, iron])
        return np.vstack(X), np.vstack(y)


# ------------------ Facade ------------------


class Spectrometer:
    def __init__(self, cfg=None):
        # allow None so code runs without a CFG.spectrometer attribute
        if cfg is None:
            class _C: pass
            cfg = _C()
            cfg.num_samples = 3
        self.cfg = cfg
        self._driver = _make_driver()
        self._model = SoilPLSModel()
        self._white_ref: Optional[np.ndarray] = None
        self._dark_ref: Optional[np.ndarray] = None

        # Pre-train with synthetic data when possible
        try:
            X, y = SoilPLSModel.make_synthetic_calibration()
            self._model.fit(X, y)
            log.info("Spectrometer ready (model pre-trained on synthetic data)")
        except Exception:
            log.warning("Pretraining skipped (sklearn missing)")

    def calibrate_white(self, n: int = 5):
        readings = [np.array(list(self._driver.read_calibrated().values())) for _ in range(n)]
        self._white_ref = np.median(readings, axis=0)
        log.info("White ref captured: mean=%.0f", self._white_ref.mean())

    def calibrate_dark(self, n: int = 5):
        readings = [np.array(list(self._driver.read_calibrated().values())) for _ in range(n)]
        self._dark_ref = np.median(readings, axis=0)
        log.info("Dark ref captured: mean=%.0f", self._dark_ref.mean())

    def _take_raw(self) -> np.ndarray:
        raw = np.array(list(self._driver.read_calibrated().values()))
        if self._dark_ref is not None:
            raw = raw - self._dark_ref
        if self._white_ref is not None:
            # scale to white reference
            with np.errstate(divide='ignore', invalid='ignore'):
                raw = raw / (self._white_ref + 1e-9) * np.mean(self._white_ref)
        return np.clip(raw, 0, None)

    def read_spectrum(self, n_samples: Optional[int] = None) -> SpectralReading:
        n = n_samples or getattr(self.cfg, "num_samples", 3)
        raws = []
        for _ in range(n):
            raws.append(self._take_raw())
        median_spec = np.median(raws, axis=0)
        temp = self._driver.read_temperature()
        channels = {wl: float(v) for wl, v in zip(sorted(WAVELENGTHS_NM), median_spec)}
        return SpectralReading(channels=channels, temperature_c=temp)

    def measure(self, n_samples: Optional[int] = None) -> SoilAnalysisResult:
        reading = self.read_spectrum(n_samples)
        arr = reading.to_array()
        preds, conf = self._model.predict(arr)

        moisture = float(np.clip(preds[0], 0, 100))
        org_matter = float(np.clip(preds[1], 0, 100))
        iron_oxide = float(np.clip(preds[2], 0, 1))

        result = SoilAnalysisResult(
            moisture_pct=round(moisture, 1),
            organic_matter_pct=round(org_matter, 1),
            iron_oxide_idx=round(iron_oxide, 3),
            confidence=conf,
            raw_spectrum=reading,
        )
        log.info("Measurement complete: %s", result.report())
        return result

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass
