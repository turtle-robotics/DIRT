"""DIRT master configuration.

Single source of truth for every tunable parameter. Import the shared
instance everywhere instead of hard-coding numbers::

    from config import CFG
    CFG.control.x_gain

Set ``PLATFORM`` below to match the hardware before running. Everything runs
in simulation by default, so no hardware is needed to try the code.

Note: the camera, marker, motor-pin and HC-SR04 sections come from the
earlier concept. The PDR design drives motors from the ESP32 through MDD10A
drivers, and ``SpectrometerConfig`` still describes an AS7265x rather than the
StellarNet EPP2000-HR. See the README's Known issues section.
"""

from dataclasses import dataclass, field
from enum import Enum


class Platform(Enum):
    """Which hardware the code talks to. Read through ``CFG.platform``."""

    RASPBERRY_PI  = "raspberry_pi"   # Full onboard vision + control
    ESP32_SERIAL  = "esp32_serial"   # Vision on laptop, commands over UART
    SIMULATION    = "simulation"     # No hardware needed – runs on any machine


# ── Change this one line to switch platforms ──────────────────────────────────
PLATFORM = Platform.SIMULATION
# ─────────────────────────────────────────────────────────────────────────────


# Camera -----------------------------------------------------------------------
@dataclass
class CameraConfig:
    """Camera used by ``core.alignment_system.CameraThread``."""

    camera_id:    int = 0
    frame_width:  int = 640
    frame_height: int = 480
    fps:          int = 30


# Marker detection (HSV colour ranges for red markers) -------------------------
@dataclass
class MarkerConfig:
    """Red-marker detection settings for ``core.alignment_system.AlignmentDetector``.

    HSV values use OpenCV ranges: hue 0..180, saturation and value 0..255.
    """

    hsv_lower_1: tuple = (0,   100, 100)
    hsv_upper_1: tuple = (10,  255, 255)
    hsv_lower_2: tuple = (170, 100, 100)   # Red wraps in HSV
    hsv_upper_2: tuple = (180, 255, 255)
    min_contour_area:   int   = 100         # px² – rejects noise blobs
    num_markers:        int   = 2
    confidence_cutoff:  float = 0.25        # Ignore detections below this


# Alignment tolerances ---------------------------------------------------------
@dataclass
class ToleranceConfig:
    """How close is close enough. The system counts as aligned when every error is under its limit.
    """

    x_pixels:         float = 20.0   # px
    y_pixels:         float = 20.0   # px
    rotation_degrees: float = 5.0    # °


# Motor GPIO pins (Raspberry Pi BCM numbering) ---------------------------------
@dataclass
class MotorPinConfig:
    """Pi GPIO pins for ``core.alignment_system.RPiMotors`` (legacy, BCM numbering)."""

    x_pin:      int = 17
    y_pin:      int = 27
    z_pin:      int = 22
    enable_pin: int = 4
    pwm_freq:   int = 1000   # Hz


# ESP32 serial (only used when PLATFORM = ESP32_SERIAL) -----------------------
@dataclass
class SerialConfig:
    """Serial link used by ``core.alignment_system.ESP32SerialMotors``."""

    port:     str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout:  float = 1.0


# Control algorithm ------------------------------------------------------------
@dataclass
class ControlConfig:
    """Proportional alignment controller settings, used by ``ControlLaw``.

    Speed for an axis is ``min(max_speed, |error| * gain)``. Raise a gain to
    correct faster; lower it if the rover overshoots and oscillates.
    """

    max_speed:            int   = 255      # PWM ceiling
    correction_threshold: float = 5.0     # px – dead-band, prevents buzz
    x_gain:               float = 2.0     # Proportional gain (tuned to avoid overshoot)
    y_gain:               float = 2.0
    z_gain:               float = 8.0     # Rotation needs higher gain (degrees, not px)
    command_duration:     float = 0.5     # s per correction pulse
    smoothing_frames:     int   = 3       # Rolling-average window for offsets
    max_iterations:       int   = 100
    timeout_s:            float = 30.0


# Distance sensor (HC-SR04) ----------------------------------------------------
@dataclass
class DistanceSensorConfig:
    """HC-SR04 ultrasonic sensor used by ``core.distance_sensor`` (legacy)."""

    trig_pin:       int   = 9
    echo_pin:       int   = 10
    max_distance_cm: float = 200.0
    sample_rate_hz: float = 10.0
    capture_duration_s: float = 5.0


# AS7265x Spectrometer ---------------------------------------------------------
@dataclass
class SpectrometerConfig:
    """Spectrometer settings used by ``spectrometer.spectrometer``.

    These describe the 18-band AS7265x, not the StellarNet EPP2000-HR the rover
    now uses. Update them along with the driver.
    """

    # I²C address (default 0x49 for AS7265x triad)
    i2c_address: int  = 0x49
    gain:        int  = 3     # 0=1x 1=3.7x 2=16x 3=64x
    integration: int  = 50    # ms per reading
    led_current: int  = 12    # mA – triad has 3 LEDs
    num_samples: int  = 5     # Average this many readings per measurement
    wavelengths: tuple = (
        410, 435, 460, 485, 510, 535,   # AS72651
        560, 585, 610, 645, 680, 705,   # AS72652
        730, 760, 810, 860, 900, 940,   # AS72653
    )


# Safety -----------------------------------------------------------------------
@dataclass
class SafetyConfig:
    """Safety limits.

    Nothing in this repository reads these yet. They are placeholders for the
    drill-depth limit, e-stop input and watchdog that the ESP32 firmware or ROS 2
    stack will enforce.
    """

    max_drill_depth_cm: float = 20.0
    emergency_stop_pin: int   = 25    # Active-low GPIO
    watchdog_timeout_s: float = 5.0


# Navigation -------------------------------------------------------------------
@dataclass
class NavigationConfig:
    """GPS serial port and waypoint settings used by ``navigation.navigator``."""

    gps_port:       str   = "/dev/ttyAMA0"
    gps_baudrate:   int   = 9600
    slam_map_res:   float = 0.05   # m/cell
    waypoint_tolerance_m: float = 0.5


# ── Assembled config object (import this everywhere) ─────────────────────────
@dataclass
class Config:
    """All config sections in one object. Use the module-level ``CFG`` instance."""

    platform:    Platform            = PLATFORM
    camera:      CameraConfig        = field(default_factory=CameraConfig)
    marker:      MarkerConfig        = field(default_factory=MarkerConfig)
    tolerance:   ToleranceConfig     = field(default_factory=ToleranceConfig)
    motor_pins:  MotorPinConfig      = field(default_factory=MotorPinConfig)
    serial:      SerialConfig        = field(default_factory=SerialConfig)
    control:     ControlConfig       = field(default_factory=ControlConfig)
    distance:    DistanceSensorConfig= field(default_factory=DistanceSensorConfig)
    spectrometer:SpectrometerConfig  = field(default_factory=SpectrometerConfig)
    safety:      SafetyConfig        = field(default_factory=SafetyConfig)
    navigation:  NavigationConfig    = field(default_factory=NavigationConfig)


CFG = Config()

