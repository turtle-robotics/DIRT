#!/usr/bin/env python3
"""DIRT display helper node (ROS 2 Jazzy, runs on the Pi 5).

Does three jobs for the touchscreen display:
  1. Sample logger: when someone taps Take sample, writes the current soil
     readings, odom position, PLS output and the full spectrum to CSV files
     in ~/dirt_samples, then publishes /dirt/samples/saved so the screen can
     mark the sample as stored on the Pi.
  2. Pi system stats: publishes CPU, memory, disk and uptime once a second on
     /dirt/pi/stats (JSON string) and the CPU temperature on
     /dirt/pi/cpu_temperature.
  3. E-stop gate (optional, off by default): sits between Nav2 and the
     ESP32. With gate_enabled:=true it passes /cmd_vel_nav through to
     /cmd_vel and sends zero velocity at 10 Hz while /dirt/estop is true.
     Remap Nav2's velocity output to /cmd_vel_nav when you turn this on.
     This is a software stop only. The physical e-stop should still cut
     motor power.

Run:  python3 dirt_display_node.py
      python3 dirt_display_node.py --ros-args -p gate_enabled:=true
"""
import csv
import datetime
import json
import shutil
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Empty, Float32, Float32MultiArray, String
from sensor_msgs.msg import Temperature
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# Keep these in sync with the SENSORS block at the top of dirt_display.html
TOPIC_SOIL_TEMP = '/dirt/soil/temperature'
TOPIC_MOISTURE = '/dirt/soil/moisture'
TOPIC_PH = '/dirt/soil/ph'
TOPIC_EC = '/dirt/soil/ec'
TOPIC_SPECTRUM = '/dirt/spectrometer/reflectance'
TOPIC_PLS = '/dirt/pls/predictions'
TOPIC_ODOM = '/odom'
TOPIC_SAMPLE_REQ = '/dirt/ui/sample_request'
TOPIC_SAVED = '/dirt/samples/saved'
TOPIC_STATS = '/dirt/pi/stats'
TOPIC_CPU_TEMP = '/dirt/pi/cpu_temperature'
TOPIC_ESTOP = '/dirt/estop'

PLS_LABELS = ['organic_carbon_pct', 'total_nitrogen_pct', 'clay_pct']
SPECTRUM_WL_START, SPECTRUM_WL_END = 400, 1000


class DirtDisplayNode(Node):
    def __init__(self):
        super().__init__('dirt_display_node')
        self.declare_parameter('sample_dir', str(Path.home() / 'dirt_samples'))
        self.declare_parameter('gate_enabled', False)
        self.declare_parameter('gate_in', '/cmd_vel_nav')
        self.declare_parameter('gate_out', '/cmd_vel')

        self.sample_dir = Path(self.get_parameter('sample_dir').value).expanduser()
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.latest = {}
        self.spectrum = None
        self.pls = None
        self.estop = False
        self.prev_cpu = self._read_cpu()

        # latest readings for the sample logger
        self.create_subscription(Temperature, TOPIC_SOIL_TEMP, lambda m: self._set('soil_temp_C', m.temperature), 10)
        self.create_subscription(Float32, TOPIC_MOISTURE, lambda m: self._set('moisture_vwc_pct', m.data), 10)
        self.create_subscription(Float32, TOPIC_PH, lambda m: self._set('ph', m.data), 10)
        self.create_subscription(Float32, TOPIC_EC, lambda m: self._set('ec_uS_cm', m.data), 10)
        self.create_subscription(Float32MultiArray, TOPIC_SPECTRUM, self._on_spectrum, 10)
        self.create_subscription(Float32MultiArray, TOPIC_PLS, self._on_pls, 10)
        self.create_subscription(Odometry, TOPIC_ODOM, self._on_odom, qos_profile_sensor_data)
        self.create_subscription(Empty, TOPIC_SAMPLE_REQ, self._on_sample_request, 10)
        self.saved_pub = self.create_publisher(String, TOPIC_SAVED, 10)

        # Pi stats
        self.stats_pub = self.create_publisher(String, TOPIC_STATS, 10)
        self.cpu_temp_pub = self.create_publisher(Temperature, TOPIC_CPU_TEMP, 10)
        self.create_timer(1.0, self._publish_stats)

        # e-stop and optional velocity gate
        self.create_subscription(Bool, TOPIC_ESTOP, self._on_estop, 10)
        self.gate = bool(self.get_parameter('gate_enabled').value)
        if self.gate:
            self.gate_pub = self.create_publisher(Twist, self.get_parameter('gate_out').value, 10)
            self.create_subscription(Twist, self.get_parameter('gate_in').value, self._on_cmd, 10)
            self.create_timer(0.1, self._hold_stop)
            self.get_logger().info('E-stop gate on: %s -> %s' % (self.get_parameter('gate_in').value, self.get_parameter('gate_out').value))
        self.get_logger().info('Saving samples to %s' % self.sample_dir)

    # ---------- sample logger ----------
    def _set(self, key, value):
        self.latest[key] = float(value)

    def _on_spectrum(self, msg):
        self.spectrum = list(msg.data)

    def _on_pls(self, msg):
        self.pls = list(msg.data)

    def _on_odom(self, msg):
        self.latest['x_m'] = msg.pose.pose.position.x
        self.latest['y_m'] = msg.pose.pose.position.y

    def _on_sample_request(self, _msg):
        now = datetime.datetime.now()
        day = now.strftime('%Y%m%d')
        samples_file = self.sample_dir / ('samples_%s.csv' % day)
        spectra_file = self.sample_dir / ('spectra_%s.csv' % day)

        fields = ['time', 'x_m', 'y_m', 'soil_temp_C', 'moisture_vwc_pct', 'ph', 'ec_uS_cm'] + PLS_LABELS + ['spectrum_bands']
        row = {k: self.latest.get(k, '') for k in fields}
        row['time'] = now.isoformat(timespec='seconds')
        for i, label in enumerate(PLS_LABELS):
            row[label] = self.pls[i] if self.pls and i < len(self.pls) else ''
        row['spectrum_bands'] = len(self.spectrum) if self.spectrum else 0

        new = not samples_file.exists()
        with samples_file.open('a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)

        if self.spectrum:
            n = len(self.spectrum)
            step = (SPECTRUM_WL_END - SPECTRUM_WL_START) / max(1, n - 1)
            new_spec = not spectra_file.exists()
            with spectra_file.open('a', newline='') as f:
                w = csv.writer(f)
                if new_spec:
                    w.writerow(['time'] + ['%.1f_nm' % (SPECTRUM_WL_START + i * step) for i in range(n)])
                w.writerow([row['time']] + ['%.5f' % v for v in self.spectrum])

        self.saved_pub.publish(String(data=json.dumps({'t': row['time'], 'file': samples_file.name, 'bands': row['spectrum_bands']})))
        self.get_logger().info('Wrote sample row to %s' % samples_file.name)

    # ---------- Pi stats ----------
    @staticmethod
    def _read_cpu():
        with open('/proc/stat') as f:
            parts = [int(x) for x in f.readline().split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        return idle, sum(parts)

    def _publish_stats(self):
        stats = {}
        try:
            idle, total = self._read_cpu()
            d_idle, d_total = idle - self.prev_cpu[0], total - self.prev_cpu[1]
            self.prev_cpu = (idle, total)
            stats['cpu'] = round(100.0 * (1 - d_idle / d_total), 1) if d_total else 0.0
        except OSError:
            pass
        try:
            info = {}
            with open('/proc/meminfo') as f:
                for line in f:
                    k, v = line.split(':', 1)
                    info[k] = int(v.split()[0])
            stats['mem'] = round(100.0 * (1 - info['MemAvailable'] / info['MemTotal']), 1)
        except (OSError, KeyError, ValueError):
            pass
        try:
            du = shutil.disk_usage(self.sample_dir)
            stats['disk'] = round(100.0 * du.used / du.total, 1)
        except OSError:
            pass
        try:
            with open('/proc/uptime') as f:
                stats['uptime'] = int(float(f.read().split()[0]))
        except OSError:
            pass
        self.stats_pub.publish(String(data=json.dumps(stats)))

        try:
            with open('/sys/class/thermal/thermal_zone0/temp') as f:
                temp_c = int(f.read().strip()) / 1000.0
            t = Temperature()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'pi5_cpu'
            t.temperature = temp_c
            self.cpu_temp_pub.publish(t)
        except (OSError, ValueError):
            pass

    # ---------- e-stop gate ----------
    def _on_estop(self, msg):
        if bool(msg.data) != self.estop:
            self.estop = bool(msg.data)
            if self.estop:
                self.get_logger().warn('E-stop engaged from %s' % TOPIC_ESTOP)
            else:
                self.get_logger().info('E-stop released')
            if self.gate and self.estop:
                self.gate_pub.publish(Twist())

    def _on_cmd(self, msg):
        self.gate_pub.publish(Twist() if self.estop else msg)

    def _hold_stop(self):
        if self.estop:
            self.gate_pub.publish(Twist())


def main():
    rclpy.init()
    node = DirtDisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
