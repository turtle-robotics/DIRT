"""
DIRT Bot – Master Configuration
================================
Single source of truth for every tunable parameter.
Set PLATFORM to match your actual hardware before running.
"""

from dataclasses import dataclass, field
from enum import Enum


class Platform(Enum):
    RASPBERRY_PI  = "raspberry_pi"   # Full onboard vision + control
    ESP32_SERIAL  = "esp32_serial"   # Vision on laptop, commands over UART
    SIMULATION    = "simulation"     # No hardware needed – runs on any machine


# ── Change this one line to switch platforms ──────────────────────────────────
PLATFORM = Platform.SIMULATION
# ─────────────────────────────────────────────────────────────────────────────


# Camera -----------------------------------------------------------------------
@dataclass
class CameraConfig:
    camera_id:    int = 0
    frame_width:  int = 640
    frame_height: int = 480
    fps:          int = 30


# Marker detection (HSV colour ranges for red markers) -------------------------
@dataclass
class MarkerConfig:
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
    x_pixels:         float = 20.0   # px
    y_pixels:         float = 20.0   # px
    rotation_degrees: float = 5.0    # °


# Motor GPIO pins (Raspberry Pi BCM numbering) ---------------------------------
@dataclass
class MotorPinConfig:
    x_pin:      int = 17
    y_pin:      int = 27
    z_pin:      int = 22
    enable_pin: int = 4
    pwm_freq:   int = 1000   # Hz


# ESP32 serial (only used when PLATFORM = ESP32_SERIAL) -----------------------
@dataclass
class SerialConfig:
    port:     str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout:  float = 1.0


# Control algorithm ------------------------------------------------------------
@dataclass
class ControlConfig:
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
    trig_pin:       int   = 9
    echo_pin:       int   = 10
    max_distance_cm: float = 200.0
    sample_rate_hz: float = 10.0
    capture_duration_s: float = 5.0


# AS7265x Spectrometer ---------------------------------------------------------
@dataclass
class SpectrometerConfig:
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
    max_drill_depth_cm: float = 20.0
    emergency_stop_pin: int   = 25    # Active-low GPIO
    watchdog_timeout_s: float = 5.0


# Navigation -------------------------------------------------------------------
@dataclass
class NavigationConfig:
    gps_port:       str   = "/dev/ttyAMA0"
    gps_baudrate:   int   = 9600
    slam_map_res:   float = 0.05   # m/cell
    waypoint_tolerance_m: float = 0.5


# ── Assembled config object (import this everywhere) ─────────────────────────
@dataclass
class Config:
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

