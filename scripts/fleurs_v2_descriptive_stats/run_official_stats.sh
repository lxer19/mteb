#!/bin/bash
# Supplies FFmpeg to TorchCodec and keeps the machine awake while it decodes
# every FLEURS `.v2` audio clip, language by language.
# The lid must stay open -- caffeinate cannot prevent a clamshell sleep.
#
# Requires a Miniforge/conda env named `fleurs-ffmpeg` with `ffmpeg` installed,
# so TorchCodec has FFmpeg's shared libraries to load.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
exec caffeinate -is env \
  DYLD_FALLBACK_LIBRARY_PATH="$HOME/miniforge3/envs/fleurs-ffmpeg/lib" \
  .venv/bin/python -u \
  scripts/fleurs_v2_descriptive_stats/rerun_official_stats.py
