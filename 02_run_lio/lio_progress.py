#!/usr/bin/env python3
"""Live progress and divergence watchdog for a LIO recording run.
Usage (env.sh sourced, same ROS_DOMAIN_ID as the LIO):
  python lio_progress.py <bag_being_played> [--vmax 4.0]
Every 5 s: % of walk processed, current speed, ETA.
Prints '!! DIVERGING' when speed stays above vmax (default 4 m/s, walking is ~1.4):
stop the run immediately, it will not recover."""
import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosbags.rosbag2 import Reader

ap = argparse.ArgumentParser()
ap.add_argument('bag')
ap.add_argument('--vmax', type=float, default=4.0)
args = ap.parse_args()
with Reader(Path(args.bag)) as r:
    START, END = r.start_time / 1e9, r.end_time / 1e9


class Progress(Node):
    def __init__(self):
        super().__init__('lio_progress')
        self.count, self.last_stamp, self.last_wall = 0, None, None
        self.hist = deque()           # (stamp, x, y, z) over the last ~2 s of walk time
        self.bad_reports = 0
        self.t0 = time.time()
        self.create_subscription(Odometry, '/odometry', self.cb, qos_profile_sensor_data)
        self.create_timer(5.0, self.report)

    def cb(self, msg):
        s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        self.count += 1
        self.last_stamp, self.last_wall = s, time.time()
        self.hist.append((s, p.x, p.y, p.z))
        while self.hist and self.hist[0][0] < s - 2.0:
            self.hist.popleft()

    def speed(self):
        if len(self.hist) < 2:
            return float('nan')
        a, b = np.array(self.hist[0]), np.array(self.hist[-1])
        dt = b[0] - a[0]
        return float(np.linalg.norm(b[1:] - a[1:]) / dt) if dt > 0.5 else float('nan')

    def report(self):
        el = time.time() - self.t0
        if self.last_stamp is None:
            print(f'[{el:5.0f}s] no /odometry yet', flush=True)
            return
        done = min(max((self.last_stamp - START) / (END - START), 0.0), 1.0)
        eta = el * (1 - done) / done if done > 0.01 else float('nan')
        v = self.speed()
        stale = time.time() - self.last_wall
        msg = (f'[{el:5.0f}s] {100 * done:5.1f}% of walk  {self.count} odom  '
               f'speed {v:5.2f} m/s  ETA {eta / 60:4.1f} min')
        if stale > 10:
            msg += f'   !! no odometry for {stale:.0f}s (finished, or LIO stalled)'
        if v == v and v > args.vmax:
            self.bad_reports += 1
            if self.bad_reports >= 2:
                msg += (f'\n   !! DIVERGING: {v:.1f} m/s is not walking. Stop this walk now '
                        f'(Ctrl+C player, recorder, LIO) and re-run it alone.')
        else:
            self.bad_reports = 0
        print(msg, flush=True)


rclpy.init()
rclpy.spin(Progress())
