#!/bin/bash
# Re-score SHARP (unfilled) with the extra source photos, plus their source features.
cd ~/dev/single-view-nvs && . .venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
set -e
nice python -m nvs.render_eval --method sharp --scenes all
nice python -m nvs.source_features > /dev/null
echo MORE_SOURCES_DONE
