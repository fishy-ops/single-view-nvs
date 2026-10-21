#!/bin/bash
# After the per-frame SDXL pass: 4-keyframe edge-aware lift (Mac) -> layered re-render (desktop) -> stability eval (Mac).
cd ~/Documents/single-view-nvs
KEYS=(m360_kitchen_src030 fortress_src021 m360_counter_src117 horns_src033 m360_bonsai_src012)
while pgrep -f "[r]un_fill_pilot.sh" >/dev/null; do sleep 30; done
nice .venv-depth/bin/python -W ignore -m nvs.fill_pilot "${KEYS[@]}" --skip-sdxl-frames --lift-v2 2>&1 | grep --line-buffered -E "keyframe|Error|Traceback"
echo LIFTV2_DONE
rsync -a results_local/fill aurora:dev/single-view-nvs/results/ && rsync -a nvs/ aurora:dev/single-view-nvs/nvs/ && \
ssh aurora "cd ~/dev/single-view-nvs && . .venv/bin/activate && PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 nice python -m nvs.render_lifted --v2 ${KEYS[*]} 2>&1 | grep -E 'lifted|Error|Traceback'" && \
rsync -a aurora:dev/single-view-nvs/results/fill/ results_local/fill/ && echo V2_RENDERED
