#!/bin/bash
# SHARP with the same Telea hole fill as the warp baselines; runs after the pair export.
cd ~/dev/single-view-nvs && . .venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
while pgrep -f "queue_pairs[.]sh|nvs[.]export_pairs" >/dev/null; do sleep 20; done
nice python -m nvs.render_eval --method sharp --fill --scenes all && echo SHARPFILL_DONE
