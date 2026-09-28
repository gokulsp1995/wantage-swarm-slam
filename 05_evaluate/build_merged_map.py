#!/usr/bin/env python3
"""Build the dense merged map from a Swarm-SLAM run.

Does NOT rely on pose_timestamps<robot>.csv -- in this cslam build that file is written
essentially empty (a per-cycle buffer that gets cleared, not a cumulative log), so there
is no reliable keyframe-to-timestamp mapping recorded anywhere on disk.

Instead, for each keyframe: take its RAW position from initial_global_pose_graph.g2o,
find the closest-matching /odometry message in that robot's own bag by position (this is
self-contained -- no dependence on any logged timestamp), take that message's own header
timestamp, then find the /cloud_registered_body scan closest to that timestamp. Transform
that scan by the keyframe's OPTIMISED pose (from optimized_global_pose_graph.g2o) and
merge everything into one PCD.

Usage:
  python build_merged_map.py --bags <walk3_odom> <walk4_odom> <walk5_odom> <walk6_odom>
                             [--results DIR] [--voxel 0.1] [--min-range 1.0] [--max-range 40]
                             [--out merged_map.pcd] [--per-robot]
--bags are given in robot order (robot 0 first). Output points are coloured by robot.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_results import decode, latest_runs, parse_g2o, snapshots  # noqa: E402

TS = get_typestore(Stores.ROS2_JAZZY)
COLORS = [(230, 25, 230), (40, 200, 40), (30, 120, 255), (255, 140, 0),
          (230, 30, 30), (0, 200, 200), (160, 100, 40), (120, 120, 120)]


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def voxel(points, size):
    if len(points) == 0 or size <= 0:
        return points
    q = np.floor(points / size).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    return points[np.sort(idx)]


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def cdr_header_stamp(raw):
    sec, nsec = struct.unpack_from('<iI', raw, 4)   # after the 4-byte CDR encapsulation
    return sec + nsec * 1e-9


def xyz_from_cloud(msg):
    off = {f.name: f.offset for f in msg.fields}
    dt = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4'] * 3,
                   'offsets': [off['x'], off['y'], off['z']], 'itemsize': msg.point_step})
    a = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
    p = np.stack([a['x'], a['y'], a['z']], axis=1).astype(np.float64)
    return p[np.isfinite(p).all(axis=1)]


def write_pcd(path, xyz, rgb):
    packed = ((rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8)
              | rgb[:, 2].astype(np.uint32)).view(np.float32)
    data = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')])
    data['x'], data['y'], data['z'], data['rgb'] = xyz[:, 0], xyz[:, 1], xyz[:, 2], packed
    head = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\n'
            'SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
            f'WIDTH {len(xyz)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyz)}\nDATA binary\n')
    with open(path, 'wb') as f:
        f.write(head.encode())
        f.write(data.tobytes())


def load_odometry(bag):
    """Return (times, positions) for every /odometry message, sorted by header time."""
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
    return t[order], p[order]


def match_keyframes_to_times(raw_positions, odom_t, odom_p, pos_tol):
    """For each keyframe's raw g2o position, find the closest /odometry message's time."""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(odom_p)
        keys = sorted(raw_positions)
        targets = np.array([raw_positions[k] for k in keys])
        d, idx = tree.query(targets)
    except ImportError:
        keys = sorted(raw_positions)
        idx, d = [], []
        for k in keys:
            diffs = np.linalg.norm(odom_p - raw_positions[k], axis=1)
            i = np.argmin(diffs)
            idx.append(i)
            d.append(diffs[i])
        idx, d = np.array(idx), np.array(d)
    times = {}
    n_far = 0
    for k, i, dist in zip(keys, idx, d):
        if dist > pos_tol:
            n_far += 1
        times[k] = odom_t[i]
    if n_far:
        print(f'    warning: {n_far}/{len(keys)} keyframes matched further than {pos_tol} m away')
    return times


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bags', nargs='+', required=True)
    ap.add_argument('--results', default=str(Path.home() / 'Projects/Autodiscovery/Swarm-SLAM/runs/results'))
    ap.add_argument('--voxel', type=float, default=0.1)
    ap.add_argument('--min-range', type=float, default=1.0, help='drop points this close (the operator)')
    ap.add_argument('--max-range', type=float, default=40.0)
    ap.add_argument('--tol', type=float, default=0.15, help='max keyframe/scan time difference (s)')
    ap.add_argument('--pos-tol', type=float, default=0.5,
                    help='warn if a keyframe/odometry position match is further than this (m)')
    ap.add_argument('--out', default='merged_map.pcd')
    ap.add_argument('--per-robot', action='store_true')
    a = ap.parse_args()

    run, robots = latest_runs(Path(a.results))
    best = None
    for rid, d in robots.items():
        snaps = snapshots(d)
        if not snaps:
            continue
        verts, _ = parse_g2o(snaps[-1] / 'optimized_global_pose_graph.g2o')
        score = (len({decode(k)[0] for k in verts}), len(verts))
        if best is None or score > best[0]:
            best = (score, snaps[-1], verts)
    if best is None:
        sys.exit('no optimised graph found')
    _, best_snap, opt_verts = best
    print(f'run {run}: using {best_snap} ({len(opt_verts)} vertices)')

    init_path = best_snap / 'initial_global_pose_graph.g2o'
    if not init_path.exists():
        sys.exit(f'{init_path} not found -- needed to match keyframes to raw odometry positions')
    raw_verts, _ = parse_g2o(init_path)

    all_xyz, all_rgb = [], []
    for rid, bag in enumerate(a.bags):
        opt_keys = sorted(k for k in opt_verts if decode(k)[0] == rid)
        raw_pos = {k: np.array(raw_verts[k][:3]) for k in opt_keys if k in raw_verts}
        if not raw_pos:
            print(f'robot {rid}: no keyframes for this robot in the initial graph, skipped')
            continue
        print(f'robot {rid} ({Path(bag).name}): matching {len(raw_pos)} keyframes by position...')
        odom_t, odom_p = load_odometry(bag)
        if len(odom_t) == 0:
            print(f'  no /odometry in {bag}, skipped')
            continue
        kf_times = match_keyframes_to_times(raw_pos, odom_t, odom_p, a.pos_tol)

        keys = sorted(kf_times)
        kt = np.array([kf_times[k] for k in keys])
        order = np.argsort(kt)
        kt, keys = kt[order], [keys[i] for i in order]
        best_dt = np.full(len(keys), np.inf)
        chosen = {}
        with Reader(Path(bag)) as r:
            conns = [c for c in r.connections if c.topic == '/cloud_registered_body']
            if not conns:
                sys.exit(f'{bag} has no /cloud_registered_body')
            for c, _, raw in r.messages(connections=conns):
                t = cdr_header_stamp(raw)
                j = np.searchsorted(kt, t)
                for cand in (j - 1, j):
                    if 0 <= cand < len(kt):
                        d = abs(kt[cand] - t)
                        if d < a.tol and d < best_dt[cand]:
                            msg = TS.deserialize_cdr(raw, c.msgtype)
                            p = xyz_from_cloud(msg)
                            rng = np.linalg.norm(p, axis=1)
                            p = voxel(p[(rng > a.min_range) & (rng < a.max_range)], a.voxel)
                            v = opt_verts[keys[cand]]
                            chosen[cand] = p @ quat_to_R(*v[3:7]).T + v[:3]
                            best_dt[cand] = d
        pts = voxel(np.concatenate(list(chosen.values())), a.voxel) if chosen else np.zeros((0, 3))
        print(f'  {len(chosen)}/{len(keys)} keyframes matched a scan, '
              f'{len(pts)} points after {a.voxel} m voxel filter')
        rgb = np.tile(np.array(COLORS[rid % len(COLORS)], dtype=np.uint8), (len(pts), 1))
        if a.per_robot and len(pts):
            out = Path(a.out).with_name(Path(a.out).stem + f'_robot{rid}.pcd')
            write_pcd(out, pts, rgb)
            print(f'  wrote {out}')
        all_xyz.append(pts)
        all_rgb.append(rgb)

    if not all_xyz or sum(len(x) for x in all_xyz) == 0:
        sys.exit('nothing to write')
    xyz, rgb = np.concatenate(all_xyz), np.concatenate(all_rgb)
    write_pcd(a.out, xyz, rgb)
    print(f'wrote {a.out}: {len(xyz)} points, coloured by robot')


if __name__ == '__main__':
    main()
