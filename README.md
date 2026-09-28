# Wantage GeoScan S2 -> Swarm-SLAM pipeline

Four independent handheld walks (Livox MID360 + IMU + RTK), each treated as
one "robot," merged with Swarm-SLAM (https://github.com/MISTLab/Swarm-SLAM)
and evaluated against RTK GPS.

## Quick start
1. `cp paths.env.example paths.env` and fill in your 3 real paths
2. Follow `00_setup/SETUP.md` once (build Swarm-SLAM, build CRUSH, venv)
3. For each walk: `01_prepare_data` -> `02_run_lio` -> `03_check_and_fix` (only if QA fails)
4. `04_run_swarm_slam` with all four walks' odometry bags
5. `05_evaluate`

See `docs/NOTES.md` for background on specific issues found along the way
(the spark_fast_lio gravity-alignment bug, heading-coupled z-drift, etc).
