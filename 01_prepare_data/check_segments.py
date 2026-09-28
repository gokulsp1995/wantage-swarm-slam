#!/usr/bin/env python3
"""Report start/end/duration of each 1-minute segment and any gaps between them.
Usage: python check_segments.py <.../raw/ros2bag>
Needs the 'rosbags' package (pip install rosbags)."""
import sys
from pathlib import Path
from rosbags.rosbag2 import Reader

root = Path(sys.argv[1])
segs = []
for d in root.glob('data_*_ros2'):
    with Reader(d) as r:
        segs.append((r.start_time / 1e9, r.end_time / 1e9, d.name))
segs.sort()
total = 0.0
for i, (s, e, name) in enumerate(segs):
    gap = s - segs[i - 1][1] if i else 0.0
    flag = '   <-- GAP' if gap > 0.5 else ''
    print(f'{name:16s} start {s:.2f}  dur {e - s:6.1f}s  gap-before {gap:6.2f}s{flag}')
    total += e - s
print(f'{len(segs)} segments, {total / 60:.1f} min of data, '
      f'span {(segs[-1][1] - segs[0][0]) / 60:.1f} min')
