#!/usr/bin/env python3
"""Cut an odometry bag down to its outbound leg only, stopping at the turnaround point.

For an out-and-back walk, the turnaround is the moment the walker is furthest (straight-
line) from the start -- distance-from-start rises before it and falls after. This finds
that point directly from /odometry rather than guessing a fixed time budget, since walks
differ in length and in how the outbound/return time splits.

Writes /odometry, /cloud_registered_body and /handsfree/rtk/gnss up to and including the
turnaround (plus --pad seconds), unchanged -- no reattachment needed, since nothing after
the cut is kept.

Usage:
  python truncate_at_turnaround.py <in_bag> <out_bag> [--pad 2.0] [--plot] [--at-seconds T]
--at-seconds overrides auto-detection (use this if the walk has a real detour so the
farthest point isn't the true turnaround -- check --plot first).
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


def find_turnaround(bag):
    t, p = [], []
    with Reader(Path(bag)) as r:
        conns = [c for c in r.connections if c.topic == '/odometry']
        for c, _, raw in r.messages(connections=conns):
            m = TS.deserialize_cdr(raw, c.msgtype)
            q = m.pose.pose.position
            t.append(stamp(m.header))
            p.append((q.x, q.y, q.z))
    t, p = np.array(t), np.array(p)
    order = np.argsort(t, kind='stable')
    t, p = t[order], p[order]
    dist = np.linalg.norm(p[:, :2] - p[0, :2], axis=1)   # horizontal distance from start
    i = np.argmax(dist)
    return t, dist, i


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('in_bag')
    ap.add_argument('out_bag')
    ap.add_argument('--pad', type=float, default=2.0, help='extra seconds kept past the turnaround')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--at-seconds', type=float, default=None,
                    help='override: cut this many seconds into the walk instead of auto-detecting')
    a = ap.parse_args()

    t, dist, i = find_turnaround(a.in_bag)
    auto_t = t[i] - t[0]
    print(f'auto-detected turnaround: {auto_t:.1f}s into the walk, {dist[i]:.1f} m from start '
          f'(walk duration {t[-1] - t[0]:.1f}s)')

    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t - t[0], dist, lw=0.8)
        ax.axvline(auto_t, color='r', ls='--', label=f'auto turnaround ({auto_t:.0f}s)')
        if a.at_seconds is not None:
            ax.axvline(a.at_seconds, color='g', ls='--', label=f'--at-seconds ({a.at_seconds:.0f}s)')
        ax.set_xlabel('seconds into walk')
        ax.set_ylabel('distance from start (m)')
        ax.legend()
        out = str(Path(a.in_bag)).rstrip('/') + '_turnaround.png'
        fig.savefig(out, dpi=120, bbox_inches='tight')
        print(f'plot saved: {out}  -- check this if the walk has real detours before trusting the cut')

    cut_t = t[0] + (a.at_seconds if a.at_seconds is not None else auto_t) + a.pad
    print(f'cutting at {cut_t - t[0]:.1f}s into the walk (including {a.pad}s padding)')

    with Reader(Path(a.in_bag)) as r, Writer(Path(a.out_bag), version=8) as w:
        conns = {c.id: w.add_connection(c.topic, c.msgtype, typestore=TS) for c in r.connections}
        kept = dropped = 0
        for c, t_bag, raw in r.messages():
            if c.topic in ('/odometry', '/cloud_registered_body', '/handsfree/rtk/gnss'):
                m = TS.deserialize_cdr(raw, c.msgtype)
                if stamp(m.header) > cut_t:
                    dropped += 1
                    continue
            w.write(conns[c.id], t_bag, raw)
            kept += 1
    print(f'\nwrote {a.out_bag}: kept {kept} messages, dropped {dropped} (everything after the turnaround)')
    print('re-run odom_qa.py to confirm; this bag now has only the outbound leg')


if __name__ == '__main__':
    main()
