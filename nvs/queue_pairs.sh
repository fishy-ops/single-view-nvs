#!/bin/bash
# Runs after nvs/queue_m360.sh finishes (waits on that script's process, not on python names).
cd ~/dev/single-view-nvs && . .venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
while pgrep -f "queue_m360[.]sh" >/dev/null; do sleep 20; done
nice python -m nvs.export_pairs && echo PAIRS_DONE
