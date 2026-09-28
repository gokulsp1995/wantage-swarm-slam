#!/usr/bin/env bash
# Merge the 1-minute ros2bag segments of one walk into a single bag,
# keeping only the topics FAST-LIO and Swarm-SLAM need.
# Usage: merge_walk.sh <.../raw/ros2bag> <output_bag_dir>
set -euo pipefail
SRC="${1:?give the ros2bag folder that holds data_0_ros2 ... data_17_ros2}"
OUT="${2:?give an output bag path (must not exist yet)}"
[ -e "$OUT" ] && { echo "Output $OUT already exists; remove it or pick another name"; exit 1; }

INPUTS=()
for d in $(ls -d "$SRC"/data_*_ros2 | sort -V); do
  [ -f "$d/metadata.yaml" ] || { echo "Skipping $d (no metadata.yaml)"; continue; }
  INPUTS+=(--input "$d")
done
echo "Merging ${#INPUTS[@]} arguments ($(( ${#INPUTS[@]} / 2 )) segments) from $SRC"

OPTS=$(mktemp --suffix=.yaml)
cat > "$OPTS" << YAML
output_bags:
- uri: $OUT
  storage_id: sqlite3
  all_topics: false
  topics: [/lidar, /imu, /handsfree/rtk/gnss]
YAML
ros2 bag convert "${INPUTS[@]}" --output-options "$OPTS"
rm -f "$OPTS"
ros2 bag info "$OUT"
