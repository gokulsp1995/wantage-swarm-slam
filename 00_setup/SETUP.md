# One-time setup

1. Build Swarm-SLAM (see swarm.repos)
2. Build CRUSH separately (not part of this repo -- clone it yourself)
3. Apply `mid360_wantage.yaml` as your spark_fast_lio config
   (note: sets common.base_frame so gravity_alignment actually runs --
   see docs/NOTES.md for why this matters)
