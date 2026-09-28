#!/usr/bin/env bash
# Play the spark_fast_lio (CRUSH) odometry bags of N walks at once, one robot namespace each.
# Usage: play_swarm.sh <rate> <odom_bag_r0> [<odom_bag_r1> ...]
set -euo pipefail
RATE="${1:?give a playback rate, e.g. 1.0}"; shift
[ $# -ge 1 ] || { echo "give at least one odometry bag"; exit 1; }
PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null' INT TERM EXIT
i=0
for BAG in "$@"; do
  echo "r$i <- $BAG"
  ros2 bag play "$BAG" -r "$RATE" --remap \
    /odometry:=/r$i/odom \
    /cloud_registered_body:=/r$i/pointcloud \
    /handsfree/rtk/gnss:=/r$i/gps/fix &
  PIDS+=($!)
  i=$((i + 1))
done
wait