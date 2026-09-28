#!/usr/bin/env python3
"""Rewrite a bag's Livox CustomMsg lidar topic as PointCloud2 in the Livox driver's
native layout (livox_ros_driver2, xfer_format=0), which spark_fast_lio's
lidar_type 5 (LIVOX_PC2) expects. All other topics are copied unchanged.

Native layout (from livox_ros_driver2 src/lddc.cpp, struct LivoxPointXyzrtlt, packed):
  x f32 @0, y f32 @4, z f32 @8, intensity f32 @12, tag u8 @16, line u8 @17,
  timestamp f64 @18  -> point_step 26
  timestamp = absolute point time in ns (timebase + offset_time)
  header.stamp = timebase (time of first point)

Usage:
  python custommsg_to_pc2.py <in_bag_dir> <out_bag_dir> [--lidar-topic /lidar]
                             [--time relative]   # only if your handler wants offsets

Needs: pip install rosbags   (no ROS workspace or livox package required)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader, Writer
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

CUSTOM_POINT = """uint32 offset_time
float32 x
float32 y
float32 z
uint8 reflectivity
uint8 tag
uint8 line
"""
CUSTOM_MSG = """std_msgs/Header header
uint64 timebase
uint32 point_num
uint8  lidar_id
uint8[3]  rsvd
CustomPoint[] points
"""
LIVOX_MSG = 'livox_ros_driver2/msg/CustomMsg'

# CDR element layout of CustomPoint inside a sequence: 19 bytes of fields,
# padded to 20 so the next element's uint32 is 4-aligned.
CDR_POINT = np.dtype({'names': ['offset_time', 'x', 'y', 'z', 'reflectivity', 'tag', 'line'],
                      'formats': ['<u4', '<f4', '<f4', '<f4', 'u1', 'u1', 'u1'],
                      'offsets': [0, 4, 8, 12, 16, 17, 18], 'itemsize': 20})
# Livox driver native PointCloud2 point (packed, 26 bytes).
NATIVE_POINT = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'tag', 'line', 'timestamp'],
                         'formats': ['<f4', '<f4', '<f4', '<f4', 'u1', 'u1', '<f8'],
                         'offsets': [0, 4, 8, 12, 16, 17, 18], 'itemsize': 26})


def align(pos, n):
    return (pos + n - 1) // n * n


def parse_custommsg(raw):
    """Fast little-endian CDR parse of livox_ros_driver2/msg/CustomMsg.
    Returns (sec, nanosec, frame_id, timebase, points_structured_array)."""
    buf = memoryview(raw)
    if bytes(buf[0:2]) != b'\x00\x01':
        raise ValueError('expected little-endian CDR (encapsulation 00 01)')
    b = buf[4:]  # CDR alignment is relative to the byte after the 4-byte encapsulation
    sec, nanosec, flen = np.frombuffer(b, dtype='<i4,<u4,<u4', count=1)[0]
    pos = 12
    frame_id = bytes(b[pos:pos + flen - 1]).decode()
    pos = align(pos + flen, 8)
    timebase = int(np.frombuffer(b, dtype='<u8', count=1, offset=pos)[0])
    pos += 8
    pos += 4 + 1 + 3                      # point_num, lidar_id, rsvd[3]
    pos = align(pos, 4)
    n = int(np.frombuffer(b, dtype='<u4', count=1, offset=pos)[0])
    pos += 4
    need = n * 20
    chunk = bytes(b[pos:pos + need])
    if len(chunk) < need:                 # last element may omit its padding byte
        chunk += b'\x00' * (need - len(chunk))
    pts = np.frombuffer(chunk, dtype=CDR_POINT, count=n)
    return int(sec), int(nanosec), frame_id, timebase, pts


def to_native(pts, timebase, relative):
    out = np.zeros(len(pts), dtype=NATIVE_POINT)
    out['x'], out['y'], out['z'] = pts['x'], pts['y'], pts['z']
    out['intensity'] = pts['reflectivity'].astype(np.float32)
    out['tag'], out['line'] = pts['tag'], pts['line']
    off = pts['offset_time'].astype(np.float64)
    out['timestamp'] = off if relative else float(timebase) + off
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--lidar-topic', default='/lidar')
    ap.add_argument('--time', choices=['absolute', 'relative'], default='absolute',
                    help='absolute = Livox driver native (default); relative = offset from timebase')
    args = ap.parse_args()

    ts = get_typestore(Stores.ROS2_JAZZY)
    ts.register(get_types_from_msg(CUSTOM_POINT, 'livox_ros_driver2/msg/CustomPoint'))
    ts.register(get_types_from_msg(CUSTOM_MSG, LIVOX_MSG))
    PointCloud2 = ts.types['sensor_msgs/msg/PointCloud2']
    PointField = ts.types['sensor_msgs/msg/PointField']
    Header = ts.types['std_msgs/msg/Header']
    Time = ts.types['builtin_interfaces/msg/Time']
    fields = [PointField(name=nm, offset=o, datatype=dt, count=1) for nm, o, dt in
              [('x', 0, 7), ('y', 4, 7), ('z', 8, 7), ('intensity', 12, 7),
               ('tag', 16, 2), ('line', 17, 2), ('timestamp', 18, 8)]]  # 7=FLOAT32 2=UINT8 8=FLOAT64

    dst = Path(args.dst)
    if dst.exists():
        sys.exit(f'{dst} already exists; remove it or choose another name')

    with Reader(Path(args.src)) as reader, Writer(dst, version=8) as writer:
        conns = {}
        for c in reader.connections:
            if c.topic == args.lidar_topic:
                if c.msgtype != LIVOX_MSG:
                    sys.exit(f'{c.topic} is {c.msgtype}, expected {LIVOX_MSG}')
                conns[c.id] = writer.add_connection(c.topic, 'sensor_msgs/msg/PointCloud2', typestore=ts)
            else:
                conns[c.id] = writer.add_connection(c.topic, c.msgtype, typestore=ts)
        total = reader.message_count
        print(f'{total} messages, lidar topic {args.lidar_topic} -> PointCloud2 ({args.time} point times)')

        done, lidar, t0, span = 0, 0, time.time(), max(reader.duration, 1)
        for c, t, raw in reader.messages():
            if c.topic == args.lidar_topic:
                sec, nsec, frame, timebase, pts = parse_custommsg(raw)
                data = to_native(pts, timebase, args.time == 'relative')
                msg = PointCloud2(
                    header=Header(stamp=Time(sec=sec, nanosec=nsec), frame_id=frame),
                    height=1, width=len(data), fields=fields, is_bigendian=False,
                    point_step=26, row_step=26 * len(data),
                    data=np.frombuffer(data.tobytes(), dtype=np.uint8), is_dense=True)
                writer.write(conns[c.id], t, ts.serialize_cdr(msg, 'sensor_msgs/msg/PointCloud2'))
                lidar += 1
            else:
                writer.write(conns[c.id], t, raw)
            done += 1
            if done % 20000 == 0:
                pct = 100.0 * (t - reader.start_time) / span
                print(f'  {pct:5.1f}%  {lidar} scans converted  {time.time() - t0:5.0f}s elapsed', flush=True)
        print(f'done: {lidar} scans converted, {done} messages written to {dst}')


if __name__ == '__main__':
    main()
