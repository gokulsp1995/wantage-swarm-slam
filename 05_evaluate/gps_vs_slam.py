#!/usr/bin/env python3
"""Compare the merged Swarm-SLAM trajectory against RTK GPS ground truth.

gps_robot_<id>.csv is keyed by the LOCAL keyframe index for that robot (0,1,2,...),
same numbering the g2o's decode() gives for that robot -- no timestamp matching needed.
Like pose_timestamps, these logs can be split/incomplete across snapshots (only the
elected optimizer writes them, and possibly not cumulatively), so every snapshot in
every run is searched and merged, keeping the newest value for any repeated keyframe.

All robots' GPS is converted to ONE shared local ENU frame (a single fixed origin --
robot 0's first fix -- not one origin per robot), matching the one shared SLAM frame
Swarm-SLAM already produces after merging. A single horizontal (x,y) rigid alignment
(rotation + translation, no scale) is then fit across all robots' matched keyframes at
once with Kabsch/Umeyama, since SLAM's frame origin and heading are arbitrary. Vertical
(z) error is reported separately, unrotated, since altitude is normally far noisier
(both from GPS and from LIO, see the walk 3 heading-coupled z-drift finding) and folding
it into a 3D rotation would let bad z corrupt the horizontal alignment.

Usage:
  python gps_vs_slam.py [--results DIR] [--out gps_vs_slam.png] [--walks 3 4 5 6]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_results import decode, latest_runs, parse_g2o  # noqa: E402

EARTH_R = 6371000.0


def load_gps(results_dir, robot):
    """Merge gps_robot_<robot>.csv across every snapshot in every run: {keyframe: (lat,lon,alt)}.
    GPS is re-logged every optimisation cycle (unlike pose_timestamps), so the same keyframe
    can appear in many snapshot files. Process snapshots in chronological order (their
    directory names are timestamps) so a later, more settled value always wins over an
    earlier one for the same keyframe -- otherwise filesystem glob order decides, which is
    arbitrary. Drops rows with NaN/Inf or a (0,0) "no fix" placeholder, which some GPS/RTK
    receivers emit before achieving lock and which would otherwise corrupt the alignment."""
    out, dropped = {}, 0
    files = sorted(Path(results_dir).glob(f'*/*/gps_robot_{robot}.csv'), key=lambda p: p.parent.name)
    for f in files:
        for line in open(f).readlines()[1:]:
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue
            kf, lat, lon, alt = parts
            try:
                lat, lon, alt = float(lat), float(lon), float(alt)
            except ValueError:
                continue
            if not (np.isfinite(lat) and np.isfinite(lon) and np.isfinite(alt)):
                dropped += 1
                continue
            if abs(lat) < 0.01 and abs(lon) < 0.01:   # (0,0) "no fix" placeholder
                dropped += 1
                continue
            out[int(kf)] = (lat, lon, alt)   # later (chronologically) files overwrite earlier ones
    if dropped:
        print(f'  robot {robot}: dropped {dropped} invalid GPS row(s) (NaN or (0,0) no-fix placeholder) '
              f'across {len(files)} snapshot files, {len(out)} unique keyframes kept')
    return out


def reject_outliers(gps_pts, robot_of, mad_k=8.0):
    """Drop points whose horizontal distance from the median position is more than mad_k
    robust median-absolute-deviations away -- a single bad RTK fix (multipath, a jump
    before lock) can sit thousands of metres off and would otherwise dominate the SVD."""
    med = np.median(gps_pts[:, :2], axis=0)
    dist = np.linalg.norm(gps_pts[:, :2] - med, axis=1)
    mad = np.median(np.abs(dist - np.median(dist))) or 1.0
    keep = dist < np.median(dist) + mad_k * mad * 1.4826
    n_bad = (~keep).sum()
    if n_bad:
        print(f'\ndropped {n_bad} outlier GPS fix(es) beyond {mad_k} MAD from the median position '
              f'(likely multipath or a jump before RTK lock)')
        for r in np.unique(robot_of[~keep]):
            print(f'  robot {r}: {(robot_of[~keep] == r).sum()} outlier(s), '
                  f'furthest {dist[~keep][robot_of[~keep] == r].max():.1f} m from median')
    return keep


def latlon_to_enu(lat, lon, alt, lat0, lon0, alt0):
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))
    east = (lon - lon0) * m_per_deg_lon
    north = (lat - lat0) * m_per_deg_lat
    up = alt - alt0
    return east, north, up


def kabsch_2d(src, dst):
    """Best-fit rotation (about z) + translation mapping src -> dst, no scale. Returns (R, t)."""
    src_c, dst_c = src.mean(0), dst.mean(0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    t = dst_c - R @ src_c
    return R, t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=str(Path.home() / 'Projects/Autodiscovery/Swarm-SLAM/runs/results'))
    ap.add_argument('--out', default='gps_vs_slam.png')
    ap.add_argument('--walks', nargs='*', type=int, default=[3, 4, 5, 6])
    ap.add_argument('--robots', nargs='*', type=int, default=None,
                    help='only include these robot IDs (e.g. --robots 0 1 2 to exclude walk 6). '
                         'Default: every robot present in the pose graph.')
    a = ap.parse_args()
    walk = {i: (a.walks[i] if i < len(a.walks) else '?') for i in range(16)}

    run, robots_dirs = latest_runs(Path(a.results))
    best = None
    for rid, d in robots_dirs.items():
        for snap in d.iterdir():
            g = snap / 'optimized_global_pose_graph.g2o'
            if g.exists():
                verts, _ = parse_g2o(g)
                score = (len({decode(k)[0] for k in verts}), len(verts))
                if best is None or score > best[0]:
                    best = (score, g, verts)
    if best is None:
        sys.exit('no optimised graph found')
    _, g2o_path, verts = best
    print(f'run {run}: using {g2o_path} ({len(verts)} vertices)')

    slam_by_robot = defaultdict(dict)   # robot -> {keyframe: (x,y,z)}
    for k, v in verts.items():
        r, kf = decode(k)
        slam_by_robot[r][kf] = np.array(v[:3])

    robots = sorted(slam_by_robot)
    if a.robots is not None:
        robots = [r for r in robots if r in a.robots]
        excluded = [r for r in slam_by_robot if r not in a.robots]
        if excluded:
            print(f'excluding robot(s) {excluded} (walk(s) {[walk[r] for r in excluded]}) per --robots')
    gps_by_robot = {r: load_gps(a.results, r) for r in robots}
    for r in robots:
        print(f'robot {r} (walk {walk[r]}): {len(gps_by_robot[r])} GPS fixes logged, '
              f'{len(slam_by_robot[r])} SLAM keyframes')

    # one shared ENU origin: robot 0's earliest-keyframe GPS fix
    if 0 not in gps_by_robot or not gps_by_robot[0]:
        sys.exit('robot 0 has no GPS data to anchor the shared ENU origin')
    k0 = min(gps_by_robot[0])
    lat0, lon0, alt0 = gps_by_robot[0][k0]
    print(f'shared ENU origin: robot 0 keyframe {k0} ({lat0:.6f}, {lon0:.6f}, {alt0:.2f} m)')

    slam_pts, gps_pts, robot_of = [], [], []
    for r in robots:
        for kf in sorted(gps_by_robot[r]):   # sort by keyframe so the plotted line follows the true path
            if kf not in slam_by_robot[r]:
                continue
            lat, lon, alt = gps_by_robot[r][kf]
            e, n, u = latlon_to_enu(lat, lon, alt, lat0, lon0, alt0)
            gps_pts.append((e, n, u))
            slam_pts.append(slam_by_robot[r][kf])
            robot_of.append(r)
    if len(slam_pts) < 3:
        sys.exit('fewer than 3 matched keyframes total -- not enough to align')
    slam_pts, gps_pts, robot_of = np.array(slam_pts), np.array(gps_pts), np.array(robot_of)
    print(f'\n{len(slam_pts)} matched keyframes total across all robots')

    keep = reject_outliers(gps_pts, robot_of)
    slam_pts, gps_pts, robot_of = slam_pts[keep], gps_pts[keep], robot_of[keep]
    print(f'{len(slam_pts)} keyframes remaining after outlier rejection')

    R, t = kabsch_2d(slam_pts[:, :2], gps_pts[:, :2])
    aligned_xy = slam_pts[:, :2] @ R.T + t
    horiz_err = np.linalg.norm(aligned_xy - gps_pts[:, :2], axis=1)
    z_err = slam_pts[:, 2] - slam_pts[:, 2].mean() - (gps_pts[:, 2] - gps_pts[:, 2].mean())

    print(f'\noverall horizontal ATE (RMSE): {np.sqrt((horiz_err**2).mean()):.2f} m')
    print(f'overall horizontal ATE (mean): {horiz_err.mean():.2f} m, max: {horiz_err.max():.2f} m')
    print('per robot:')
    for r in robots:
        m = robot_of == r
        if m.sum() == 0:
            print(f'  robot {r} (walk {walk[r]}): no matched keyframes')
            continue
        print(f'  robot {r} (walk {walk[r]}): {m.sum()} points, '
              f'horizontal ATE RMSE {np.sqrt((horiz_err[m]**2).mean()):.2f} m, '
              f'z error RMSE {np.sqrt((z_err[m]**2).mean()):.2f} m (unrotated, see note above)')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', 3: 'tab:red',
                  4: 'tab:purple', 5: 'tab:brown', 6: 'tab:pink', 7: 'tab:gray'}
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.plot(gps_pts[:, 0], gps_pts[:, 1], color='black', lw=1.5, ls='--', label='GPS', zorder=1)
        for r in robots:
            m = robot_of == r
            if m.sum() == 0:
                continue
            order = np.argsort([kf for kf, rr in zip(
                [kf for rr2 in [r] for kf in slam_by_robot[rr2]], [r] * 0)] or range(m.sum()))
            ax.plot(aligned_xy[m, 0], aligned_xy[m, 1], color=colors.get(r, 'k'), lw=1.2,
                    label=f'Robot {r} (walk {walk[r]})', zorder=2)
        ax.set_aspect('equal')
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.legend()
        ax.set_title('Swarm-SLAM trajectory estimates vs. RTK GPS ground truth')
        fig.savefig(a.out, dpi=150, bbox_inches='tight')
        print(f'\nplot saved: {a.out}')
    except ImportError:
        print('\n(matplotlib not installed: pip install matplotlib to get the plot)')


if __name__ == '__main__':
    main()