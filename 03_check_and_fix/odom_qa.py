#!/usr/bin/env python3
"""Quality check for LIO odometry bags before feeding them to Swarm-SLAM.

Detects the failure that makes a robot "launch like a rocket": LIO divergence,
where scan matching fails ("No Effective Points!") and the filter runs on the
IMU alone, so speed grows without bound.

Usage (Swarm terminal, env.sh sourced):
  python odom_qa.py <odom_bag> [<odom_bag> ...] [--source <pc2_bag> ...] [--vmax 4.0] [--plot]

  --source   the converted input bag(s) (same order) to also check lidar gaps,
             expected scan count and whether the walk starts stationary
  --vmax     max plausible speed in m/s (default 4.0: walking is ~1.4, jogging ~3)
  --plot     save <odom_bag>_qa.png with the trajectory and the divergence point
Exit code 0 if every bag passes, 1 otherwise.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_JAZZY)


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def read_odometry(bag):
    t, p, clouds = [], [], 0
    with Reader(Path(bag)) as r:
        for c, _, raw in r.messages():
            if c.topic == '/odometry':
                m = TS.deserialize_cdr(raw, c.msgtype)
                q = m.pose.pose.position
                t.append(stamp(m.header))
                p.append((q.x, q.y, q.z))
            elif c.topic == '/cloud_registered_body':
                clouds += 1
    t, p = np.array(t), np.array(p)
    order = np.argsort(t, kind='stable')
    return t[order], p[order], clouds


def windowed_speed(t, p, window=1.0):
    """Speed over ~1 s windows, robust to the jitter of 200 Hz odometry."""
    j = np.searchsorted(t, t + window)
    ok = j < len(t)
    i, j = np.nonzero(ok)[0], j[ok]
    dt = t[j] - t[i]
    good = dt > 0.5 * window
    i, j, dt = i[good], j[good], dt[good]
    return i, np.linalg.norm(p[j] - p[i], axis=1) / dt


def check_source(bag):
    lidar_t, imu = [], []
    with Reader(Path(bag)) as r:
        for c, _, raw in r.messages():
            if c.topic == '/lidar':
                lidar_t.append(stamp(TS.deserialize_cdr(raw, c.msgtype).header))
            elif c.topic == '/imu' and len(imu) < 2000:
                m = TS.deserialize_cdr(raw, c.msgtype)
                a, g = m.linear_acceleration, m.angular_velocity
                imu.append((stamp(m.header), a.x, a.y, a.z, g.x, g.y, g.z))
    lidar_t = np.sort(np.array(lidar_t))
    gaps = np.diff(lidar_t)
    imu = np.array(imu)
    first = imu[imu[:, 0] < imu[0, 0] + 2.0] if len(imu) else imu
    return {
        'scans': len(lidar_t),
        'max_gap': float(gaps.max()) if len(gaps) else 0.0,
        'gaps_over_0.3s': int((gaps > 0.3).sum()),
        'gap_times': (lidar_t[:-1][gaps > 0.3] - lidar_t[0])[:5],
        'start_gyro': float(np.linalg.norm(first[:, 4:7], axis=1).mean()) if len(first) else float('nan'),
        'start_acc_std': float(np.linalg.norm(first[:, 1:4], axis=1).std()) if len(first) else float('nan'),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--source', nargs='*', default=[])
    ap.add_argument('--vmax', type=float, default=4.0)
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()
    if a.source and len(a.source) != len(a.bags):
        sys.exit('--source needs one input bag per odometry bag, in the same order')

    all_ok = True
    for k, bag in enumerate(a.bags):
        print(f'\n=== {bag}')
        t, p, clouds = read_odometry(bag)
        if len(t) < 10:
            print('  FAIL: almost no /odometry messages'); all_ok = False; continue
        dur = t[-1] - t[0]
        i, v = windowed_speed(t, p)
        over = v > a.vmax
        steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
        path = steps.sum()
        horiz_gap = np.linalg.norm(p[-1, :2] - p[0, :2])
        print(f'  duration {dur:7.1f} s   odometry {len(t)} msgs   clouds {clouds}')
        print(f'  path {path:8.1f} m   mean speed {path / dur:5.2f} m/s   '
              f'max 1s-speed {v.max():7.2f} m/s')
        print(f'  max single step {steps.max():.2f} m   '
              f'closure gap horiz {horiz_gap:.2f} m  z {p[-1, 2] - p[0, 2]:+.2f} m')
        ok = True
        if over.any():
            first = i[np.argmax(over)]
            print(f'  FAIL: DIVERGENCE - speed exceeds {a.vmax} m/s from '
                  f'{t[first] - t[0]:.1f} s into the walk '
                  f'({100 * over.mean():.1f}% of the walk above the limit)')
            ok = False
        if a.source:
            s = check_source(a.source[k])
            ratio = clouds / max(s['scans'], 1)
            print(f'  input: {s["scans"]} scans (recorded {100 * ratio:.1f}%), '
                  f'max lidar gap {s["max_gap"]:.2f} s, gaps >0.3 s: {s["gaps_over_0.3s"]}')
            if s['gaps_over_0.3s']:
                print(f'    first gaps at {np.round(s["gap_times"], 1)} s into the walk')
            still = s['start_gyro'] < 0.05 and s['start_acc_std'] < 0.2
            print(f'  start: gyro {s["start_gyro"]:.3f} rad/s, accel std {s["start_acc_std"]:.3f} '
                  f'-> {"stationary" if still else "MOVING at start (bad for LIO initialisation)"}')
            if ratio < 0.98:
                print('  FAIL: more than 2% of scans missing from the odometry bag'); ok = False
        print('  PASS' if ok else '  -> re-record this walk (see runbook)')
        all_ok &= ok

        if a.plot:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 2, figsize=(14, 6))
                ax[0].plot(p[:, 0], p[:, 1], lw=0.8)
                ax[0].plot(*p[0, :2], 'go', label='start')
                if over.any():
                    f = i[np.argmax(over)]
                    ax[0].plot(*p[f, :2], 'rx', ms=12, mew=3, label='divergence')
                ax[0].set_aspect('equal'); ax[0].legend(); ax[0].set_title('x-y')
                ax[1].plot(t[i] - t[0], v, lw=0.6)
                ax[1].axhline(a.vmax, color='r', ls='--')
                ax[1].set_xlabel('s into walk'); ax[1].set_ylabel('1 s speed (m/s)')
                out = str(Path(bag)).rstrip('/') + '_qa.png'
                fig.savefig(out, dpi=110, bbox_inches='tight')
                print(f'  plot: {out}')
            except ImportError:
                print('  (matplotlib not installed: pip install matplotlib for --plot)')
    print('\nALL PASS' if all_ok else '\nSOME BAGS FAILED')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
