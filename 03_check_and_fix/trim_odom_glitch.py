#!/usr/bin/env python3
"""Cut a brief divergence glitch out of an odometry bag and rejoin the pieces.

Finds windows where 1s-speed exceeds --vmax (same detector as odom_qa.py), using each
message's HEADER timestamp (sensor time) -- not the bag's own write time, which can be
completely different if the bag was recorded long after the data was captured. Drops
/odometry and /cloud_registered_body messages inside [start-pad, end+pad] for each
glitch (matched by header stamp for both topics), then rigidly shifts every kept
/odometry pose AFTER a cut (translation + yaw only) so it reattaches exactly where the
trajectory left off before the cut. RTK is copied through untouched.

This does not recover the true path during the glitch (that data is gone) -- it removes
a bad spike so the rest of the walk isn't corrupted by carrying its position error
forward for the remainder of the recording.

Usage:
  python trim_odom_glitch.py <in_bag> <out_bag> [--vmax 4.0] [--pad 2.0] [--dry-run]
--dry-run only reports the glitch windows found and writes nothing.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_JAZZY)


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def find_glitches(t, p, vmax, window=1.0):
    j = np.searchsorted(t, t + window)
    ok = j < len(t)
    i, j = np.nonzero(ok)[0], j[ok]
    dt = t[j] - t[i]
    good = dt > 0.5 * window
    i, j, dt = i[good], j[good], dt[good]
    v = np.linalg.norm(p[j] - p[i], axis=1) / dt
    bad = v > vmax
    windows, k = [], 0
    while k < len(bad):
        if bad[k]:
            s = k
            while k < len(bad) and bad[k]:
                k += 1
            windows.append((t[i[s]], t[j[k - 1]]))
        else:
            k += 1
    return windows


def merge_windows(windows, gap=0.0):
    if not windows:
        return []
    windows = sorted(windows)
    merged = [list(windows[0])]
    for s, e in windows[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(w) for w in merged]


def yaw_from_quat(qx, qy, qz, qw):
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def quat_from_yaw(yaw):
    return 0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))


def rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('in_bag')
    ap.add_argument('out_bag')
    ap.add_argument('--vmax', type=float, default=4.0)
    ap.add_argument('--pad', type=float, default=2.0, help='extra seconds cut on each side of a glitch')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    with Reader(Path(a.in_bag)) as r:
        odom_t, odom_p = [], []
        for c, _, raw in r.messages():
            if c.topic == '/odometry':
                m = TS.deserialize_cdr(raw, c.msgtype)
                q = m.pose.pose.position
                odom_t.append(stamp(m.header))
                odom_p.append((q.x, q.y, q.z))
    odom_t, odom_p = np.array(odom_t), np.array(odom_p)
    order = np.argsort(odom_t, kind='stable')
    odom_t, odom_p = odom_t[order], odom_p[order]

    raw_windows = find_glitches(odom_t, odom_p, a.vmax)
    if not raw_windows:
        sys.exit('no glitch found above --vmax: nothing to trim, the bag already passes odom_qa.py')
    padded = [(max(odom_t[0], s - a.pad), min(odom_t[-1], e + a.pad)) for s, e in raw_windows]
    cuts = merge_windows(padded)
    print(f'found {len(cuts)} glitch window(s) (header time, seconds into the walk):')
    for s, e in cuts:
        print(f'  {s - odom_t[0]:.1f}s to {e - odom_t[0]:.1f}s ({e - s:.1f}s cut, including {a.pad}s padding each side)')
    if a.dry_run:
        return

    def in_a_cut(t):
        return any(s <= t <= e for s, e in cuts)

    last_good_pos, last_good_yaw = None, None
    pending_shift = None
    just_cut = False

    with Reader(Path(a.in_bag)) as r, Writer(Path(a.out_bag), version=8) as w:
        conns = {c.id: w.add_connection(c.topic, c.msgtype, typestore=TS) for c in r.connections}
        kept = dropped = 0
        for c, t, raw in r.messages():
            if c.topic in ('/odometry', '/cloud_registered_body'):
                m = TS.deserialize_cdr(raw, c.msgtype)   # need this for every message now: stamp lives inside
                ts = stamp(m.header)
                if in_a_cut(ts):
                    dropped += 1
                    just_cut = True
                    continue
                if c.topic == '/odometry':
                    pos = np.array([m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z])
                    q = m.pose.pose.orientation
                    yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

                    if just_cut and last_good_pos is not None:
                        dyaw = last_good_yaw - yaw
                        dpos = last_good_pos - (rotz(dyaw) @ pos)
                        pending_shift = (dpos, dyaw)
                        just_cut = False

                    if pending_shift is not None:
                        dpos, dyaw = pending_shift
                        pos = rotz(dyaw) @ pos + dpos
                        yaw = yaw + dyaw
                        qx, qy, qz, qw = quat_from_yaw(yaw)
                        m.pose.pose.position.x = float(pos[0])
                        m.pose.pose.position.y = float(pos[1])
                        m.pose.pose.position.z = float(pos[2])
                        m.pose.pose.orientation.x = qx
                        m.pose.pose.orientation.y = qy
                        m.pose.pose.orientation.z = qz
                        m.pose.pose.orientation.w = qw
                        raw = TS.serialize_cdr(m, c.msgtype)

                    last_good_pos, last_good_yaw = pos, yaw
            w.write(conns[c.id], t, raw)
            kept += 1
    print(f'\nwrote {a.out_bag}: kept {kept} messages, dropped {dropped} '
          f'(/odometry and /cloud_registered_body inside the cut windows)')
    print('re-run odom_qa.py on the trimmed bag to confirm it now passes')


if __name__ == '__main__':
    main()