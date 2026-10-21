#!/bin/bash
# Desktop queue for the Mip-NeRF 360 scenes. Waits for any running nvs python job first.
cd ~/dev/single-view-nvs && . .venv/bin/activate
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
while pgrep -f "^python -m nvs\." >/dev/null; do sleep 15; done
set -e
nice python -m nvs.render_eval --method sharp --scenes m360
nice python -m nvs.source_features --scenes m360_bicycle m360_bonsai m360_counter m360_garden m360_kitchen m360_room m360_stump > /dev/null
echo M360_SHARP_DONE
nice python -m nvs.make_warp --depth sharp --scenes m360
nice python -m nvs.make_warp --depth depthpro --scenes m360
for m in warp_sharpdepth warp_depthpro; do
  nice python -m nvs.render_eval --method $m --scenes m360
  nice python -m nvs.render_eval --method $m --scenes m360 --fill
done
echo QUEUE_DONE
