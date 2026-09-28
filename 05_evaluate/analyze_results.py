#!/usr/bin/env python3
"""Summarise a Swarm-SLAM run from its results folders.

Vertex keys are gtsam LabeledSymbols: robot = ((key >> 48) & 0xFF) - ord('A'),
keyframe = key & 0xFFFFFFFFFFFF (from cslam gtsam_utils.h: ROBOT_LABEL(id) = 'A' + id).

Usage:
  python analyze_results.py [results_dir] [--walks 3 4 5 6]
Defaults to ~/Projects/Autodiscovery/Swarm-SLAM/runs/results and the most recent run.
Prints, for the most complete optimised graph: robots present, odometry / intra-robot /
inter-robot edge counts, which robot pairs connected and at which keyframes, per-robot
trajectory extent, and a RUNAWAY flag for any robot whose optimised trajectory jumps.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

KEY_MASK = (1 << 48) - 1


def decode(key):
    return ((key >> 48) & 0xFF) - ord('A'), key & KEY_MASK


def parse_g2o(path):
    verts, edges = {}, []
    with open(path) as f:
        for line in f:
            s = line.split()
            if not s:
                continue
            if s[0].startswith('VERTEX_SE3'):
                verts[int(s[1])] = np.array([float(v) for v in s[2:9]])
            elif s[0].startswith('EDGE_SE3'):
                edges.append((int(s[1]), int(s[2]), np.array([float(v) for v in s[3:6]])))
    return verts, edges


def read_log(path):
    out = {}
    if path.exists():
        for line in open(path):
            k, _, v = line.strip().partition(',')
            if k and k != 'error':
                out[k] = v
    return out


def latest_runs(results):
    runs = defaultdict(dict)
    for d in results.iterdir():
        m = re.match(r'(.+)_experiment_robot_(\d+)$', d.name)
        if m and d.is_dir():
            runs[m.group(1)][int(m.group(2))] = d
    if not runs:
        raise SystemExit(f'no *_experiment_robot_N folders in {results}')
    key = sorted(runs)[-1]
    return key, runs[key]


def snapshots(robot_dir):
    return sorted(p for p in robot_dir.iterdir() if (p / 'optimized_global_pose_graph.g2o').exists())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results', nargs='?',
                    default=str(Path.home() / 'Projects/Autodiscovery/Swarm-SLAM/runs/results'))
    ap.add_argument('--walks', nargs='*', type=int, default=[3, 4, 5, 6],
                    help='walk number for robot 0, 1, 2, ... (default 3 4 5 6)')
    ap.add_argument('--runaway-step', type=float, default=5.0,
                    help='flag a robot whose consecutive keyframes are further apart than this (m)')
    a = ap.parse_args()
    walk = {i: (a.walks[i] if i < len(a.walks) else '?') for i in range(16)}

    run, robots = latest_runs(Path(a.results))
    print(f'run {run}: result folders for robots {sorted(robots)}')

    best = None
    for rid, d in sorted(robots.items()):
        snaps = snapshots(d)
        if not snaps:
            print(f'  robot {rid} (walk {walk[rid]}): no optimised graph written')
            continue
        verts, _ = parse_g2o(snaps[-1] / 'optimized_global_pose_graph.g2o')
        present = sorted({decode(k)[0] for k in verts})
        log = read_log(snaps[-1] / 'log.csv')
        print(f'  robot {rid} (walk {walk[rid]}): {len(snaps)} snapshots, latest {snaps[-1].name}: '
              f'{len(verts)} vertices, robots in graph {present}, '
              f'origin {log.get("origin_robot_id", "?")}, '
              f'matches ok/failed {log.get("total_nb_successful_matches", "?")}/'
              f'{log.get("total_nb_failed_matches", "?")}, total_error {log.get("total_error", "?")}')
        score = (len(present), len(verts))
        if best is None or score > best[0]:
            best = (score, rid, snaps[-1])

    if best is None:
        raise SystemExit('no optimised graphs found')
    _, rid, snap = best
    verts, edges = parse_g2o(snap / 'optimized_global_pose_graph.g2o')
    print(f'\nMost complete merged graph: robot {rid}, {snap}')

    per_robot = defaultdict(list)
    for k, v in verts.items():
        r, kf = decode(k)
        per_robot[r].append((kf, v[:3]))

    odo, intra, inter = defaultdict(int), defaultdict(int), defaultdict(list)
    for k1, k2, dxyz in edges:
        (r1, f1), (r2, f2) = decode(k1), decode(k2)
        if r1 == r2:
            (odo if abs(f1 - f2) == 1 else intra)[r1] += 1
        else:
            pair = tuple(sorted((r1, r2)))
            kf = (f1, f2) if r1 < r2 else (f2, f1)
            inter[pair].append((kf, float(np.linalg.norm(dxyz))))

    print('\nPer robot (optimised trajectory):')
    for r in sorted(per_robot):
        kv = sorted(per_robot[r], key=lambda x: x[0])
        pos = np.array([p for _, p in kv])
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=1) if len(pos) > 1 else np.zeros(1)
        ext = pos.max(0) - pos.min(0)
        flag = '  <-- RUNAWAY: odometry diverged, re-record this walk' if steps.max() > a.runaway_step else ''
        print(f'  robot {r} (walk {walk[r]}): {len(kv)} keyframes, extent '
              f'{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} m, path {steps.sum():.0f} m, '
              f'max keyframe step {steps.max():.2f} m, odometry edges {odo[r]}, '
              f'intra-robot loop closures {intra[r]}{flag}')

    print('\nInter-robot connections (edges between different robots):')
    rs = sorted(per_robot)
    if not inter:
        print('  none: the robots were never merged')
    for i, r1 in enumerate(rs):
        for r2 in rs[i + 1:]:
            e = inter.get((r1, r2), [])
            if e:
                k1 = [x[0][0] for x in e]
                k2 = [x[0][1] for x in e]
                print(f'  robot {r1}-{r2} (walks {walk[r1]}-{walk[r2]}): {len(e)} edges, '
                      f'keyframes {min(k1)}-{max(k1)} <-> {min(k2)}-{max(k2)}, '
                      f'median edge length {np.median([x[1] for x in e]):.2f} m')
            else:
                print(f'  robot {r1}-{r2} (walks {walk[r1]}-{walk[r2]}): not connected')
    missing = sorted(set(robots) - set(per_robot))
    if missing:
        print(f'\nRobots not in the merged graph: {missing} (no accepted loop closure to the others)')
    print('\nEvery edge between two robots is an accepted inter-robot loop closure (cslam creates '
          'cross-robot factors only for successful matches). Walks that start at the office are '
          'expected to connect near keyframe 0.')


if __name__ == '__main__':
    main()
