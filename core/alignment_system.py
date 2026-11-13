"""DIRT vision alignment system.

Uses a camera to find two red markers, measures how far their midpoint is from
the centre of the frame (X/Y in pixels) and how much the line between them is
rotated (degrees), then drives the X, Y and Z (rotation) motors until all three
errors are inside the tolerances in ``config.ToleranceConfig``.

Pipeline, one iteration of ``AlignmentControlSystem._loop``:

1. ``CameraThread`` keeps the latest frame in memory.
2. ``AlignmentDetector.detect`` turns the frame into an ``AlignmentData``.
3. ``ControlLaw.compute`` turns that into a ``MotorCommand``.
4. A motor backend (simulated, Pi GPIO, or ESP32 over serial) runs the command.

Note: this module is from the earlier camera/red-marker concept. The PDR
design drives the motors from the ESP32 through MDD10A drivers, so check it
against the current hardware before relying on it (see the README's Known
issues section).
"""

from __future__ import annotations

import logging
import math
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

from config import CFG, Platform

log = logging.getLogger(__name__)


@dataclass
class AlignmentData:
    """Result of one marker detection, after smoothing.

    Attributes:
        x_offset: Horizontal error in pixels, marker midpoint minus frame centre.
            Positive means the markers are right of centre.
        y_offset: Vertical error in pixels. Positive means below centre (image
            rows grow downward).
        rotation_angle: Angle of the line from marker 1 to marker 2, folded into
            -90..+90 degrees. 0 means the markers are level.
        confidence: 0..1 score based on total marker area (bigger blobs are
            more trustworthy).
        is_aligned: True when all three errors are inside tolerance.
        marker_centres: The two raw (x, y) marker centres in pixels, unsmoothed.
    """

    x_offset: float
    y_offset: float
    rotation_angle: float
    confidence: float
    is_aligned: bool
    marker_centres: list


AlignmentState = AlignmentData


class MotorDirection(Enum):
    """Direction for one motor axis. The value is used as a sign (+1, -1, 0)."""

    FORWARD = 1
    BACKWARD = -1
    STOP = 0


@dataclass
class MotorCommand:
    """One timed move for the three alignment axes.

    Speeds are on a 0..255 scale (the same scale as ``ControlConfig.max_speed``)
    and ``duration`` is in seconds. Every backend runs the move for ``duration``
    and then stops.
    """

    x_dir: MotorDirection
    x_speed: float
    y_dir: MotorDirection
    y_speed: float
    z_dir: MotorDirection
    z_speed: float
    duration: float


class SystemState(Enum):
    """States reported by ``AlignmentControlSystem.status()``."""

    IDLE = auto()
    INITIALISING = auto()
    DETECTING = auto()
    CORRECTING = auto()
    ALIGNED = auto()
    ERROR = auto()


class CameraThread(threading.Thread):
    """Background thread that reads the camera as fast as it can.

    The control loop only ever wants the newest frame, so this thread overwrites a
    single slot instead of queueing frames. That keeps latency low even if the
    control loop is slow.
    """

    def __init__(self, cfg=CFG.camera):
        """Set up the thread. The camera itself is opened in ``run``.

        Args:
            cfg: Camera settings (id, resolution, fps). Defaults to ``CFG.camera``.
        """
        super().__init__(daemon=True)
        self.cfg = cfg
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.cap: Optional[cv2.VideoCapture] = None

    def run(self):
        """Open the camera and keep the latest frame until ``stop`` is called."""
        self.cap = cv2.VideoCapture(self.cfg.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        log.info("Camera thread started: %dx%d @ %dfps",
                 self.cfg.frame_width, self.cfg.frame_height, self.cfg.fps)
        while not self._stop_event.is_set():
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
        if self.cap is not None:
            self.cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the newest frame, or None if no frame has arrived yet."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        """Ask the thread to exit. Call ``join()`` afterward to wait for it."""
        self._stop_event.set()


class AlignmentDetector:
    """Finds the two red markers in a frame and measures the alignment error.

    Readings are averaged over the last ``ControlConfig.smoothing_frames`` frames
    to reduce jitter, so the detector keeps state between calls.
    """

    def __init__(self, cfg=CFG):
        """Read settings and set up the smoothing buffers.

        Args:
            cfg: The full config object. Defaults to the global ``CFG``.
        """
        self.cam_cfg = cfg.camera
        self.mkr_cfg = cfg.marker
        self.tol = cfg.tolerance
        self.ctrl = cfg.control
        self.cx = cfg.camera.frame_width / 2.0
        self.cy = cfg.camera.frame_height / 2.0

        n = cfg.control.smoothing_frames
        self._xs: deque = deque(maxlen=n)
        self._ys: deque = deque(maxlen=n)
        self._rs: deque = deque(maxlen=n)

    def detect(self, frame: np.ndarray) -> Optional[AlignmentData]:
        """Detect the markers in one BGR frame.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            Smoothed ``AlignmentData``, or None if fewer than two markers were found
            or the confidence is below ``MarkerConfig.confidence_cutoff``.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._red_mask(hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.mkr_cfg.min_contour_area]
        if len(contours) < 2:
            log.debug("Only %d contours - need at least 2", len(contours))
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        centres = self._centres(contours)
        if len(centres) < 2:
            return None

        c1, c2 = centres[0], centres[1]
        mid_x = (c1[0] + c2[0]) / 2.0
        mid_y = (c1[1] + c2[1]) / 2.0
        raw_x = mid_x - self.cx
        raw_y = mid_y - self.cy
        raw_rot = self._normalise_marker_angle(
            math.degrees(math.atan2(c2[1] - c1[1], c2[0] - c1[0]))
        )

        # Confidence grows with blob size and saturates at 1.0 once the two
        # markers cover 1500 px^2 together. 1500 is an empirical constant tuned
        # for the default 640x480 frame; retune it if the resolution changes.
        total_area = sum(cv2.contourArea(c) for c in contours)
        confidence = min(1.0, total_area / 1500.0)

        if confidence < self.mkr_cfg.confidence_cutoff:
            log.debug("Confidence %.2f below cutoff %.2f - skipping",
                      confidence, self.mkr_cfg.confidence_cutoff)
            return None

        # Moving average over the last N detections to smooth out jitter.
        self._xs.append(raw_x)
        self._ys.append(raw_y)
        self._rs.append(raw_rot)
        x_off = float(np.mean(self._xs))
        y_off = float(np.mean(self._ys))
        rot = float(np.mean(self._rs))

        aligned = (
            abs(x_off) < self.tol.x_pixels and
            abs(y_off) < self.tol.y_pixels and
            abs(rot) < self.tol.rotation_degrees
        )
        return AlignmentData(x_off, y_off, rot, confidence, aligned, [c1, c2])

    def annotate(self, frame: np.ndarray, data: Optional[AlignmentData]) -> np.ndarray:
        """Draw the crosshair, markers and error readout on a copy of the frame.

        Args:
            frame: The frame that was passed to ``detect``.
            data: The result of ``detect`` (may be None).

        Returns:
            A new annotated image. The input frame is not modified.
        """
        out = frame.copy()
        cx, cy = int(self.cx), int(self.cy)

        cv2.line(out, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
        cv2.line(out, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)

        if data is None:
            cv2.putText(out, "NO MARKERS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return out

        for (mx, my) in data.marker_centres:
            cv2.circle(out, (int(mx), int(my)), 8, (0, 0, 255), 2)

        mid = (int(self.cx + data.x_offset), int(self.cy + data.y_offset))
        cv2.circle(out, mid, 5, (255, 0, 0), -1)
        cv2.line(out, (cx, cy), mid, (255, 100, 0), 1)

        colour = (0, 255, 0) if data.is_aligned else (0, 200, 255)
        lines = [
            f"X: {data.x_offset:+.1f}px",
            f"Y: {data.y_offset:+.1f}px",
            f"R: {data.rotation_angle:+.1f}deg",
            f"Conf: {data.confidence:.2f}",
            "ALIGNED" if data.is_aligned else "CORRECTING",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(out, txt, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)
        return out

    def _red_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Return a binary mask of red pixels.

        Red wraps around the ends of the OpenCV hue range (0..180), so two HSV ranges
        are combined. An open removes speckle noise and a close fills small holes.
        """
        m = self.mkr_cfg
        lo1, hi1 = np.array(m.hsv_lower_1), np.array(m.hsv_upper_1)
        lo2, hi2 = np.array(m.hsv_lower_2), np.array(m.hsv_upper_2)
        mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    @staticmethod
    def _normalise_marker_angle(deg: float) -> float:
        """Fold an angle into -90..+90 degrees.

        The two markers look the same, so a line at 170 degrees is the same as one at
        -10 degrees. Folding stops the rotation error from flipping sign when the
        detector swaps which marker it calls "first".
        """
        deg = deg % 180.0
        if deg > 90.0:
            deg -= 180.0
        return deg

    @staticmethod
    def _centres(contours) -> list:
        """Return the centroid (x, y) of each contour, skipping zero-area ones."""
        centres = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] > 0:
                centres.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        return centres


class _MotorBackend:
    """Interface every motor backend implements.

    ``execute`` must block for the command's duration and then stop. ``enable``
    and ``disable`` are optional and do nothing by default.
    """

    def execute(self, cmd: MotorCommand):
        """Run a ``MotorCommand``, blocking until it finishes."""
        raise NotImplementedError

    def stop(self):
        """Stop all axes immediately."""
        raise NotImplementedError

    def enable(self):
        """Power up the drivers, if the backend has an enable line."""
        pass

    def disable(self):
        """Power down the drivers, if the backend has an enable line."""
        pass


class SimulatedMotors(_MotorBackend):
    """Motor backend that only logs commands. Used when ``PLATFORM`` is SIMULATION."""

    def __init__(self):
        """Start with every axis stopped."""
        self.state = {"x": 0, "y": 0, "z": 0}

    def execute(self, cmd: MotorCommand):
        """Log the signed speed for each axis, wait for the duration, then stop."""
        self.state["x"] = cmd.x_dir.value * cmd.x_speed
        self.state["y"] = cmd.y_dir.value * cmd.y_speed
        self.state["z"] = cmd.z_dir.value * cmd.z_speed
        log.info("[SIM] Motors: X=%+.0f  Y=%+.0f  Z=%+.0f  dur=%.2fs",
                 self.state["x"], self.state["y"], self.state["z"], cmd.duration)
        time.sleep(cmd.duration)
        self.stop()

    def stop(self):
        """Zero every axis."""
        self.state = {"x": 0, "y": 0, "z": 0}
        log.info("[SIM] All motors stopped")


class RPiMotors(_MotorBackend):
    """Motor backend that drives PWM pins directly from Raspberry Pi GPIO.

    Legacy: the PDR design drives the motors from the ESP32 instead. Note that
    this backend only sets PWM duty and does not drive any direction pins, so the
    ``*_dir`` fields of a command are ignored here.
    """

    def __init__(self, cfg=CFG.motor_pins):
        """Set up the GPIO pins and start PWM at 0% duty.

        Args:
            cfg: Pin numbers and PWM frequency. Defaults to ``CFG.motor_pins``.

        Raises:
            RuntimeError: If ``RPi.GPIO`` cannot be imported (not on a Pi).
        """
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
        except ImportError as exc:
            raise RuntimeError("RPi.GPIO not available - are you on a Pi?") from exc

        self._cfg = cfg
        GPIO = self._GPIO
        GPIO.setmode(GPIO.BCM)
        for pin in (cfg.x_pin, cfg.y_pin, cfg.z_pin, cfg.enable_pin):
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        self._pwm_x = GPIO.PWM(cfg.x_pin, cfg.pwm_freq)
        self._pwm_y = GPIO.PWM(cfg.y_pin, cfg.pwm_freq)
        self._pwm_z = GPIO.PWM(cfg.z_pin, cfg.pwm_freq)
        for pwm in (self._pwm_x, self._pwm_y, self._pwm_z):
            pwm.start(0)
        log.info("[RPi] GPIO motors initialised")

    def enable(self):
        """Drive the enable pin high so the motor drivers respond."""
        self._GPIO.output(self._cfg.enable_pin, self._GPIO.HIGH)

    def disable(self):
        """Stop all axes and drive the enable pin low."""
        self.stop()
        self._GPIO.output(self._cfg.enable_pin, self._GPIO.LOW)

    def execute(self, cmd: MotorCommand):
        """Set each axis's PWM duty from its speed, wait for the duration, then stop."""
        def _duty(speed):
            """Convert a 0..255 speed to a 0..100 % duty cycle."""
            return (speed / 255.0) * 100.0

        self._pwm_x.ChangeDutyCycle(_duty(cmd.x_speed))
        self._pwm_y.ChangeDutyCycle(_duty(cmd.y_speed))
        self._pwm_z.ChangeDutyCycle(_duty(cmd.z_speed))
        time.sleep(cmd.duration)
        self.stop()

    def stop(self):
        """Set every PWM channel to 0% duty."""
        for pwm in (self._pwm_x, self._pwm_y, self._pwm_z):
            pwm.ChangeDutyCycle(0)

    def __del__(self):
        """Release the GPIO pins when the object is garbage collected."""
        try:
            self._GPIO.cleanup()
        except Exception:
            pass


class ESP32SerialMotors(_MotorBackend):
    """Motor backend that sends fixed-size command packets to the ESP32 over serial.

    Packet format, 10 unsigned bytes (``struct`` format ``!10B``)::

        [0] 0xAA header
        [1] X direction + 1   (0 = backward, 1 = stop, 2 = forward)
        [2] X speed 0..255
        [3] Y direction + 1
        [4] Y speed
        [5] Z direction + 1
        [6] Z speed
        [7] duration in ms, high byte
        [8] duration in ms, low byte
        [9] 0x55 footer

    The ESP32 firmware must parse the same layout. Durations are limited to
    65535 ms by the two duration bytes.
    """

    HEADER = 0xAA
    FOOTER = 0x55

    def __init__(self, cfg=CFG.serial):
        """Open the serial port.

        Args:
            cfg: Port, baud rate and timeout. Defaults to ``CFG.serial``.
        """
        import serial

        self._ser = serial.Serial(cfg.port, cfg.baudrate, timeout=cfg.timeout)
        log.info("[ESP32] Serial port %s @ %d baud", cfg.port, cfg.baudrate)

    def execute(self, cmd: MotorCommand):
        """Send one command packet and wait for it to finish.

        The extra 50 ms of sleep gives the ESP32 time to finish the move before the
        next packet arrives.
        """
        dur_ms = int(cmd.duration * 1000)
        payload = struct.pack(
            "!10B",
            self.HEADER,
            int(cmd.x_dir.value + 1),
            int(cmd.x_speed),
            int(cmd.y_dir.value + 1),
            int(cmd.y_speed),
            int(cmd.z_dir.value + 1),
            int(cmd.z_speed),
            (dur_ms >> 8) & 0xFF,
            dur_ms & 0xFF,
            self.FOOTER,
        )
        self._ser.write(payload)
        time.sleep(cmd.duration + 0.05)

    def stop(self):
        """Send a packet with every axis set to stop (direction byte 1) and speed 0."""
        stop_cmd = struct.pack("!10B", self.HEADER, 1, 0, 1, 0, 1, 0, 0, 0, self.FOOTER)
        self._ser.write(stop_cmd)

    def __del__(self):
        """Close the serial port when the object is garbage collected."""
        try:
            self._ser.close()
        except Exception:
            pass


def _build_motor_backend() -> _MotorBackend:
    """Return the motor backend that matches ``CFG.platform``.

    Raises:
        ValueError: If the platform is not one of the known values.
    """
    p = CFG.platform
    if p == Platform.SIMULATION:
        return SimulatedMotors()
    if p == Platform.RASPBERRY_PI:
        return RPiMotors()
    if p == Platform.ESP32_SERIAL:
        return ESP32SerialMotors()
    raise ValueError(f"Unknown platform: {p}")


class ControlLaw:
    """Proportional controller that turns alignment error into a motor command.

    For each axis: if the error is under ``correction_threshold`` the axis stays
    still (a dead band, so the motors do not chatter around zero). Otherwise the
    speed is ``|error| * gain``, capped at ``max_speed``, in the direction of the
    error's sign.
    """

    def __init__(self, cfg=CFG.control, tol=CFG.tolerance):
        """Store the controller settings.

        Args:
            cfg: Gains, dead band and speed limit. Defaults to ``CFG.control``.
            tol: Alignment tolerances. Defaults to ``CFG.tolerance``.
        """
        self.cfg = cfg
        self.tol = tol

    def compute(self, data: AlignmentData) -> Optional[MotorCommand]:
        """Compute the next correction.

        Args:
            data: The latest detection.

        Returns:
            A ``MotorCommand``, or None if the system is already aligned.
        """
        if data.is_aligned:
            return None

        def _axis(error: float, gain: float):
            """Return (direction, speed) for one axis using the dead band and gain."""
            if abs(error) < self.cfg.correction_threshold:
                return MotorDirection.STOP, 0.0
            direction = MotorDirection.FORWARD if error > 0 else MotorDirection.BACKWARD
            speed = min(self.cfg.max_speed, abs(error) * gain)
            return direction, speed

        x_dir, x_spd = _axis(data.x_offset, self.cfg.x_gain)
        y_dir, y_spd = _axis(data.y_offset, self.cfg.y_gain)
        z_dir, z_spd = _axis(data.rotation_angle, self.cfg.z_gain)

        return MotorCommand(x_dir, x_spd, y_dir, y_spd, z_dir, z_spd, self.cfg.command_duration)


class AlignmentControlSystem:
    """Runs the full detect-and-correct loop in a background thread.

    Typical use::

        with AlignmentControlSystem(show_video=True) as acs:
            acs.start()
            while acs.running:
                time.sleep(0.1)
            print(acs.status())

    The loop ends when the markers are aligned, when ``ControlConfig.timeout_s``
    or ``max_iterations`` is reached (state becomes ERROR), or when ``stop`` is
    called.
    """

    def __init__(self, show_video: bool = False):
        """Create the detector, controller, motor backend and camera thread.

        Args:
            show_video: If True, show the annotated camera feed in an OpenCV window.
                Press q in that window to end the loop.
        """
        self.show_video = show_video
        self.state = SystemState.IDLE
        self.detector = AlignmentDetector()
        self.control = ControlLaw()
        self.motors = _build_motor_backend()
        self._cam_thread = CameraThread()
        self._ctrl_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.last_alignment: Optional[AlignmentData] = None
        self.running = False

    def start(self):
        """Start the camera, enable the motors and launch the control loop.

        Waits 0.5 s after starting the camera so the first frames can arrive.
        """
        self.state = SystemState.INITIALISING
        self._cam_thread.start()
        time.sleep(0.5)
        self.motors.enable()
        self._stop_event.clear()
        self._ctrl_thread = threading.Thread(target=self._loop, daemon=False, name="alignment-ctrl")
        self._ctrl_thread.start()
        self.running = True
        log.info("Alignment system started [platform=%s]", CFG.platform.value)

    def start_alignment(self):
        """Alias for ``start`` kept for older callers."""
        self.start()

    def stop(self):
        """Stop the loop and camera, disable the motors, and close any video window."""
        self._stop_event.set()
        if self._ctrl_thread:
            self._ctrl_thread.join(timeout=CFG.control.timeout_s)
        self._cam_thread.stop()
        self._cam_thread.join(timeout=2.0)
        self.motors.disable()
        if self.show_video:
            cv2.destroyAllWindows()
        self.state = SystemState.IDLE
        self.running = False

    def stop_alignment(self):
        """Alias for ``stop`` kept for older callers."""
        self.stop()

    def __enter__(self):
        """Allow use as a context manager. Does not call ``start``."""
        return self

    def __exit__(self, *_):
        """Always call ``stop`` when leaving the ``with`` block."""
        self.stop()

    def _loop(self):
        """Detect, check, correct, repeat. Runs in the control thread (see class docs).
        """
        cfg = CFG.control
        deadline = time.monotonic() + cfg.timeout_s
        iteration = 0

        while not self._stop_event.is_set():
            if time.monotonic() > deadline:
                log.warning("Alignment timed out after %.1fs", cfg.timeout_s)
                self.state = SystemState.ERROR
                break
            if iteration >= cfg.max_iterations:
                log.warning("Reached max iterations (%d)", cfg.max_iterations)
                self.state = SystemState.ERROR
                break

            frame = self._cam_thread.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            self.state = SystemState.DETECTING
            data = self.detector.detect(frame)
            self.last_alignment = data

            if self.show_video:
                annotated = self.detector.annotate(frame, data)
                cv2.imshow("DIRT - Alignment", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if data is None:
                log.debug("No detection - waiting")
                time.sleep(0.05)
                continue

            log.info("It=%d  X=%+.1fpx  Y=%+.1fpx  R=%+.1fdeg  conf=%.2f",
                     iteration, data.x_offset, data.y_offset, data.rotation_angle, data.confidence)

            if data.is_aligned:
                self.state = SystemState.ALIGNED
                log.info("Aligned in %d iterations", iteration)
                self.motors.stop()
                break

            cmd = self.control.compute(data)
            if cmd:
                self.state = SystemState.CORRECTING
                self.motors.execute(cmd)

            iteration += 1

        self.running = False

    def status(self) -> dict:
        """Return the current state and last reading as a JSON-friendly dict."""
        a = self.last_alignment
        return {
            "state": self.state.name,
            "platform": CFG.platform.value,
            "x_offset": round(a.x_offset, 2) if a else None,
            "y_offset": round(a.y_offset, 2) if a else None,
            "rotation": round(a.rotation_angle, 2) if a else None,
            "confidence": round(a.confidence, 3) if a else None,
            "aligned": a.is_aligned if a else False,
        }

    def get_status(self) -> dict:
        """Alias for ``status`` kept for older callers."""
        return self.status()

