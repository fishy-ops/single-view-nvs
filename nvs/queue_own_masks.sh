#!/bin/bash
# Generalisation test: Depth Pro warp scored with masks from its own geometry. Runs after the re-scoring queue.
cd ~/dev/single-view-nvs && . .venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
while pgrep -f "queue_more_source[s].sh" >/dev/null; do sleep 20; done
nice python -m nvs.render_eval --method warp_depthpro --fill --own-masks --scenes all && echo OWN_DONE
