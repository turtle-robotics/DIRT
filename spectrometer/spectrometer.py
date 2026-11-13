"""DIRT spectrometer module: AS7265x driver, PLS pipeline, and mock backend.

This module provides:

* ``SpectralReading`` / ``SoilAnalysisResult`` dataclasses
* ``AS7265xDriver`` (I2C) with a ``MockAS7265xDriver`` fallback
* ``SoilPLSModel`` for training and prediction (scikit-learn, with a numpy
  least-squares fallback)
* ``Spectrometer``, a facade that ties driver -> preprocessing -> model

Data flow for one ``Spectrometer.measure()`` call::

    driver.read_calibrated()  x N    (18 channel values each)
      -> dark subtract / white scale  (if calibrated)
      -> median of the N reads
      -> Savitzky-Golay smooth -> SNV normalise
      -> PLS predict -> moisture %, organic matter %, iron oxide index
      -> Hotelling T^2 confidence

The implementation focuses on being runnable in SIMULATION mode without
hardware.

Important: the rover now uses a StellarNet EPP2000-HR (900 to 1075 nm,
2048 pixels), not the 18-band AS7265x this driver targets. The preprocessing
and PLS parts carry over; the driver and ``WAVELENGTHS_NM`` do not. The model
is also pre-trained on synthetic data only, so its predictions are
placeholders until it is trained on real calibration samples.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from pathlib import Path
from types import SimpleNamespace

# Allow running this module directly (for quick debugging) by making the
# project root available on sys.path if `config` can't be imported normally.
try:
    from config import CFG, Platform
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    """One spectrum.

    Attributes:
        channels: Map of wavelength in nm to calibrated intensity.
        timestamp: Unix time the reading was made.
        temperature_c: Sensor temperature in degrees C.
    """

    channels: Dict[int, float]
    timestamp: float = field(default_factory=time.time)
    temperature_c: float = 25.0

    def to_array(self) -> np.ndarray:
        """Return the channel values as a 1-D array, sorted by wavelength."""
        return np.array([self.channels[wl] for wl in sorted(self.channels)])

    def __repr__(self) -> str:
        """Show every channel as ``<nm>nm:<value>``."""
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
        """Create the mock driver.

        Args:
            soil_type: ``"loam"``, ``"sand"`` or ``"clay"``.
        """
        self._mock = MockAS7265xDriver(soil_type=soil_type)

    def capture(self) -> SpectrumReading:
        """Return one mock reading with its temperature."""
        channels = self._mock.read_calibrated()
        temp = self._mock.read_temperature()
        return SpectrumReading(channels=channels, temperature_c=temp)


@dataclass
class SoilAnalysisResult:
    """Soil properties predicted from one reading.

    Attributes:
        moisture_pct: Soil moisture, 0..100 %.
        organic_matter_pct: Organic matter, 0..100 %.
        iron_oxide_idx: Unitless iron oxide index, 0..1.
        confidence: 0..1, how similar the spectrum is to the training data
            (see ``SoilPLSModel._hotelling_confidence``). Low means the model is
            extrapolating.
        raw_spectrum: The reading the prediction came from.
    """

    moisture_pct: float
    organic_matter_pct: float
    iron_oxide_idx: float
    confidence: float
    raw_spectrum: SpectralReading

    def report(self) -> str:
        """Return a one-line human-readable summary."""
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
        """Open the I2C bus and configure the sensor.

        Args:
            cfg: Object with ``i2c_address``, ``gain``, ``integration``,
                ``led_current``, ``num_samples`` and ``wavelengths``. If None, safe
                defaults are used.
            bus_id: I2C bus number (1 on a Raspberry Pi).

        Raises:
            RuntimeError: If ``smbus2`` is not installed.
        """
        try:
            import smbus2
        except Exception as exc:  # pragma: no cover - hardware
            raise RuntimeError("smbus2 required for AS7265xDriver") from exc

        self._bus = smbus2.SMBus(bus_id)
        # cfg may be None when running on non-hardware machines; provide defaults
        if cfg is None:
            cfg = SimpleNamespace()
            cfg.i2c_address = AS7265X_ADDR
            cfg.gain = 1
            cfg.integration = 1
            cfg.led_current = 1
            cfg.num_samples = 3
            cfg.wavelengths = WAVELENGTHS_NM
        self._addr = getattr(cfg, "i2c_address", AS7265X_ADDR)
        self._cfg = cfg
        # perform initial configuration (gain, integration, leds)
        try:
            self._configure()
        except Exception:
            log.exception("AS7265x configuration failed")

    def _wait_for_tx(self):
        """Wait for the TX_VALID status bit before writing.

        Polls the status register every 1 ms, up to about 200 ms.

        Check against the datasheet: in the AS726x/AS7265x mailbox protocol the host
        normally waits for TX_VALID to be *clear* (write buffer empty) before writing,
        but this loop returns when it is *set*.

        Raises:
            TimeoutError: If the device never becomes ready.
        """
        for _ in range(200):
            status = self._bus.read_byte_data(self._addr, AS7265X_STATUS_REG)
            if status & AS7265X_TX_VALID:
                return
            time.sleep(0.001)
        raise TimeoutError("AS7265x TX timeout")

    def _wait_for_rx(self):
        """Wait until a byte is ready to read (RX_VALID bit set).

        Polls every 1 ms, up to about 200 ms.

        Raises:
            TimeoutError: If no data arrives.
        """
        for _ in range(200):
            status = self._bus.read_byte_data(self._addr, AS7265X_STATUS_REG)
            if status & AS7265X_RX_VALID:
                return
            time.sleep(0.001)
        raise TimeoutError("AS7265x RX timeout")

    def _write_virt(self, reg: int, val: int):
        """Write one virtual register.

        The AS7265x hides its real registers behind a small I2C mailbox: write the
        register address with bit 7 set (meaning "write"), then write the value, each
        time waiting for the device to be ready.
        """
        self._wait_for_tx()
        self._bus.write_byte_data(self._addr, AS7265X_WRITE_REG, reg | 0x80)
        self._wait_for_tx()
        self._bus.write_byte_data(self._addr, AS7265X_WRITE_REG, val)

    def _read_virt(self, reg: int) -> int:
        """Read one virtual register: write its address (bit 7 clear), wait, then read the value.
        """
        self._wait_for_tx()
        self._bus.write_byte_data(self._addr, AS7265X_WRITE_REG, reg)
        self._wait_for_rx()
        return self._bus.read_byte_data(self._addr, AS7265X_READ_REG)

    def _configure(self):
        """Write gain, integration time and LED current. Errors are logged, not raised.
        """
        cfg = self._cfg
        # These register addresses follow the common usage in drivers
        try:
            self._write_virt(0x04, cfg.gain)
            self._write_virt(0x05, cfg.integration)
            self._write_virt(0x07, cfg.led_current)
        except Exception:
            log.debug("Configuration writes failed (device may be absent)")

    def _trigger(self):
        """Start a measurement and wait up to about 500 ms for data.

        Raises:
            TimeoutError: If the measurement does not finish.
        """
        self._write_virt(0x04, AS7265X_TAKE_MEAS)
        for _ in range(500):
            status = self._bus.read_byte_data(self._addr, AS7265X_STATUS_REG)
            if status & AS7265X_RX_VALID:
                return
            time.sleep(0.001)
        raise TimeoutError("AS7265x measurement timeout")

    def read_calibrated(self) -> Dict[int, float]:
        """Take a measurement and return all 18 calibrated channels.

        Each channel is a 4-byte big-endian float starting at virtual register 0x14.
        A channel that fails to read is logged and returned as 0.0.

        Returns:
            Map of wavelength in nm to calibrated value.
        """
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
                log.warning("AS7265x channel %dnm read failed; returning 0.0", wl, exc_info=True)
                result[wl] = 0.0
        return result

    def read_temperature(self) -> float:
        """Return the device temperature in degrees C (virtual register 0x06)."""
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

    def __init__(self, soil_type: str = "loam", scale: float = 10_000.0, rng: Optional[np.random.Generator] = None):
        """Pick a base spectrum for the soil type.

        Args:
            soil_type: ``"loam"``, ``"sand"`` or ``"clay"``. Unknown names use loam.
            scale: Multiplier that turns 0..1 reflectance into sensor-like counts.
            rng: Random generator, so tests can pass a seeded one.
        """
        self._base = self._SOIL_BASES.get(soil_type, self._SOIL_BASES["loam"])
        self._scale = scale
        self._rng = rng or np.random.default_rng()
        log.info("[AS7265x] Mock driver active (soil_type=%s)", soil_type)

    def read_calibrated(self) -> Dict[int, float]:
        """Return the base spectrum plus about 1% Gaussian noise, scaled to counts."""
        noise = self._rng.normal(0, 0.01, len(WAVELENGTHS_NM))
        counts = (self._base + noise) * self._scale
        counts = np.clip(counts, 0, None)
        return {wl: float(c) for wl, c in zip(WAVELENGTHS_NM, counts)}

    def read_temperature(self) -> float:
        """Return about 25 degrees C with a little noise."""
        return 25.0 + float(self._rng.normal(0, 0.5))


def _make_driver():
    """Return the mock in simulation, otherwise try the real driver and fall back to the mock.
    """
    if CFG.platform == Platform.SIMULATION:
        return MockAS7265xDriver()
    try:
        return AS7265xDriver()
    except Exception as exc:
        log.warning("AS7265x hardware failed (%s) - using mock", exc)
        return MockAS7265xDriver()


# ------------------ Preprocessing helpers ------------------


def snv(spectrum: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: subtract the mean and divide by the standard deviation.

    This removes overall brightness and scale differences between readings (from
    sample distance, particle size or lighting) so the model sees spectral shape
    rather than how bright the sample was. 1e-9 avoids divide-by-zero.
    """
    mu = spectrum.mean()
    std = spectrum.std()
    return (spectrum - mu) / (std + 1e-9)


def savitzky_golay(spectrum: np.ndarray, window: int = 5, poly: int = 2) -> np.ndarray:
    """Smooth a spectrum with a Savitzky-Golay filter.

    For each point, fit a polynomial of degree ``poly`` to the ``window`` points
    around it and replace the point with the fitted value. This smooths noise but
    keeps peak shapes better than a plain moving average. The ``window // 2``
    points at each end are left unchanged.

    Args:
        spectrum: 1-D array.
        window: Odd number of points per fit.
        poly: Polynomial degree, less than ``window``.

    For long spectra (like the 2048-pixel StellarNet), ``scipy.signal.savgol_filter``
    does the same thing much faster.
    """
    half = window // 2
    out = spectrum.copy()
    for i in range(half, len(spectrum) - half):
        segment = spectrum[i - half: i + half + 1]
        x = np.arange(-half, half + 1)
        coeffs = np.polyfit(x, segment, poly)
        out[i] = np.polyval(coeffs, 0)
    return out


def preprocess(spectrum: np.ndarray) -> np.ndarray:
    """Standard pipeline: Savitzky-Golay smoothing, then SNV. Used for both training and prediction.
    """
    return snv(savitzky_golay(spectrum))


# ------------------ PLS model pipeline ------------------


class SoilPLSModel:
    """Partial Least Squares regression from spectrum to soil properties.

    PLS finds a few combined "components" of the spectrum that best explain the
    targets, which works well when there are many correlated wavelengths and
    relatively few samples.

    Targets, in column order, are listed in ``TARGET_NAMES``. If scikit-learn is
    missing, a plain least-squares linear model is used instead so the
    simulation still runs.
    """

    TARGET_NAMES = ["moisture_pct", "organic_matter_pct", "iron_oxide_idx"]

    def __init__(self, n_components: int = 6):
        """Create an untrained model.

        Args:
            n_components: Number of PLS components. More can fit better but
                overfits faster; tune with cross-validation on real data.
        """
        try:
            from sklearn.cross_decomposition import PLSRegression  # noqa: F401
            from sklearn.preprocessing import StandardScaler  # noqa: F401
        except Exception:
            log.warning("scikit-learn not available: model training/prediction disabled")
            # Provide a lightweight numpy fallback so the simulation still works
            self._pls = None
            self._scaler = None
            self.n_components = n_components
            self._trained = False
            self._X_train = None
            self._coeffs = None
            self._X_mean = None
            self._X_std = None
            self._cov_inv = None
            return

        from sklearn.cross_decomposition import PLSRegression
        from sklearn.preprocessing import StandardScaler

        self.n_components = n_components
        self._pls = PLSRegression(n_components=n_components)
        self._scaler = StandardScaler()
        self._trained = False
        self._X_train: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None

    def fit(self, X_raw: np.ndarray, y: np.ndarray):
        """Train on raw spectra.

        Args:
            X_raw: Shape (n_samples, n_wavelengths), raw spectra. They are
                preprocessed and standardised here, so do not preprocess first.
            y: Shape (n_samples, 3), targets in ``TARGET_NAMES`` order.
        """
        X_pre = np.apply_along_axis(preprocess, 1, X_raw)
        # If sklearn is available use the PLS pipeline
        if self._pls is not None and self._scaler is not None:
            X_sc = self._scaler.fit_transform(X_pre)
            self._pls.fit(X_sc, y)
            self._X_train = X_sc
            self._trained = True
            self._cache_covariance()
            log.info("PLS model fitted: %d samples, %d components", len(X_raw), self.n_components)
            return

        # Numpy fallback: standardize and fit linear least-squares for each target
        X_mean = X_pre.mean(axis=0)
        X_std = X_pre.std(axis=0) + 1e-9
        X_sc = (X_pre - X_mean) / X_std
        # Add intercept column
        X_aug = np.hstack([np.ones((X_sc.shape[0], 1)), X_sc])
        # Solve least-squares for each target column
        coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        # coeffs shape: (n_features+1, n_targets)
        self._coeffs = coeffs
        self._X_mean = X_mean
        self._X_std = X_std
        self._X_train = X_sc
        self._trained = True
        self._cache_covariance()
        log.info("Fallback linear model fitted: %d samples", len(X_raw))

    def _cache_covariance(self):
        """Precompute the inverse covariance matrix used for Hotelling's T^2
        confidence so we don't redo an O(features^3) inversion on every
        single prediction call."""
        if self._X_train is None:
            self._cov_inv = None
            return
        cov = np.cov(self._X_train.T) + np.eye(self._X_train.shape[1]) * 1e-6
        self._cov_inv = np.linalg.inv(cov)

    def predict(self, spectrum: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict soil properties for one raw spectrum.

        Args:
            spectrum: 1-D raw spectrum with the same wavelengths as the training data.

        Returns:
            (predictions, confidence): predictions is an array in ``TARGET_NAMES``
            order; confidence is 0..1.

        Raises:
            RuntimeError: If the model is not trained, or the spectrum length does
                not match the training data (fallback model).
        """
        if not self._trained:
            raise RuntimeError("Model not trained")

        x_pre = preprocess(spectrum)
        # If we have sklearn PLS, use it
        if self._pls is not None and self._scaler is not None:
            x_pre = x_pre.reshape(1, -1)
            x_sc = self._scaler.transform(x_pre)
            y_hat = self._pls.predict(x_sc)[0]
            conf = self._hotelling_confidence(x_sc)
            return y_hat, conf

        # Numpy fallback prediction using least-squares coeffs
        if self._coeffs is None:
            raise RuntimeError("Fallback model coefficients missing; model not trained")

        # Ensure statistics from training exist and have compatible shape
        if self._X_mean is None or self._X_std is None:
            raise RuntimeError("Fallback model stats missing; model not trained for numpy fallback")
        x_pre_arr = np.asarray(x_pre).ravel()
        mean_arr = np.asarray(self._X_mean).ravel()
        std_arr = np.asarray(self._X_std).ravel()
        if x_pre_arr.shape[0] != mean_arr.shape[0] or x_pre_arr.shape[0] != std_arr.shape[0]:
            raise RuntimeError(
                f"Fallback prediction shape mismatch: spectrum_len={x_pre_arr.shape[0]} mean_len={mean_arr.shape[0]} std_len={std_arr.shape[0]}"
            )
        x_sc = (x_pre_arr - mean_arr) / (std_arr + 1e-9)
        # Make augmented vector: [1.0, x_sc...] as 1D float array
        x_aug = np.concatenate([np.array([1.0], dtype=float), np.asarray(x_sc).ravel().astype(float)])

        try:
            y_hat = x_aug @ self._coeffs
        except Exception as exc:
            raise RuntimeError(
                f"Fallback prediction failed: x_aug.shape={x_aug.shape} coeffs_shape={getattr(self._coeffs, 'shape', None)} -> {exc}"
            ) from exc

        conf = self._hotelling_confidence(x_sc.reshape(1, -1))
        return y_hat, conf

    def _hotelling_confidence(self, x_sc: np.ndarray) -> float:
        """Score how close a sample is to the training data, 0..1.

        Computes the squared Mahalanobis distance (Hotelling's T^2) from the training
        mean, which accounts for how wavelengths vary together, then maps it to
        ``exp(-0.05 * T^2)``. A spectrum like the training set scores near 1; an
        unusual one scores near 0. The 0.05 factor is a hand-picked scale, not a
        statistical threshold.
        """
        if self._X_train is None:
            return 0.0
        if self._cov_inv is None:
            self._cache_covariance()
            if self._cov_inv is None:
                return 0.0
        mean = self._X_train.mean(axis=0)
        diff = x_sc[0] - mean
        maha2 = float(diff @ self._cov_inv @ diff)
        conf = float(np.exp(-0.05 * maha2))
        return round(min(1.0, max(0.0, conf)), 3)

    @staticmethod
    def make_synthetic_calibration(n: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        """Make fake training data from the mock soil spectra.

        Args:
            n: Number of samples, cycling through loam, sand and clay.

        Returns:
            (X, y) with X of shape (n, 18) and y of shape (n, 3).

        The targets are random numbers that do not depend on the spectra, so a model
        trained on this only proves the pipeline runs. Note the spectrum noise uses
        the global ``np.random`` rather than the seeded ``rng``, so X differs from run
        to run.
        """
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
    """Main spectrometer API used by ``main.py``.

    Example::

        spec = Spectrometer()
        spec.calibrate_dark()     # with the light blocked
        spec.calibrate_white()    # with a white reference tile
        result = spec.measure()
        print(result.report())

    The model is pre-trained on synthetic data when the object is created.
    """

    def __init__(self, cfg=None):
        """Create the driver and model, and pre-train on synthetic data.

        Args:
            cfg: Object with ``num_samples`` and ``wavelengths``. If None, defaults
                are used (``CFG.spectrometer`` is not read automatically).
        """
        # allow None so code runs without a CFG.spectrometer attribute
        if cfg is None:
            cfg = SimpleNamespace()
            cfg.num_samples = 3
            cfg.wavelengths = WAVELENGTHS_NM
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
        """Capture the white reference as the median of ``n`` reads.

        Point the sensor at a white reference tile first. Later readings are scaled
        by it so results do not depend on the lamp or sensor response.
        """
        readings = [np.array(list(self._driver.read_calibrated().values())) for _ in range(n)]
        self._white_ref = np.median(readings, axis=0)
        log.info("White ref captured: mean=%.0f", self._white_ref.mean())

    def calibrate_dark(self, n: int = 5):
        """Capture the dark reference as the median of ``n`` reads.

        Block all light first. This is the sensor's offset and is subtracted from
        later readings.
        """
        readings = [np.array(list(self._driver.read_calibrated().values())) for _ in range(n)]
        self._dark_ref = np.median(readings, axis=0)
        log.info("Dark ref captured: mean=%.0f", self._dark_ref.mean())

    def _take_raw(self) -> np.ndarray:
        """Take one read, then apply the dark and white references if they exist.

        Result is ``(raw - dark) / white * mean(white)``, clipped at 0, which keeps
        the values in counts rather than 0..1 reflectance.
        """
        raw = np.array(list(self._driver.read_calibrated().values()))
        if self._dark_ref is not None:
            raw = raw - self._dark_ref
        if self._white_ref is not None:
            # scale to white reference
            with np.errstate(divide='ignore', invalid='ignore'):
                raw = raw / (self._white_ref + 1e-9) * np.mean(self._white_ref)
        return np.clip(raw, 0, None)

    def read_spectrum(self, n_samples: Optional[int] = None) -> SpectralReading:
        """Take several reads and return their median.

        Args:
            n_samples: Number of reads. Defaults to ``cfg.num_samples``.
        """
        n = n_samples or getattr(self.cfg, "num_samples", 3)
        raws = []
        for _ in range(n):
            raws.append(self._take_raw())
        median_spec = np.median(raws, axis=0)
        temp = self._driver.read_temperature()
        channels = {wl: float(v) for wl, v in zip(sorted(WAVELENGTHS_NM), median_spec)}
        return SpectralReading(channels=channels, temperature_c=temp)

    def measure(self, n_samples: Optional[int] = None) -> SoilAnalysisResult:
        """Read a spectrum and predict soil properties.

        Predictions are clipped to valid ranges (0..100 % and 0..1).

        Args:
            n_samples: Number of reads to take the median of.
        """
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
        """Allow use as a context manager."""
        return self

    def __exit__(self, *_):
        """Nothing to clean up; present so ``with`` works."""
        pass