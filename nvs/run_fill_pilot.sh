#!/bin/bash
# Generative-fill pilot chain: LaMa + 3D lift (Mac) -> lifted re-render (desktop) -> per-frame SDXL (Mac).
cd ~/Documents/single-view-nvs
KEYS=(m360_kitchen_src030 fortress_src021 m360_counter_src117 horns_src033 m360_bonsai_src012)
nice .venv-depth/bin/python -W ignore -m nvs.fill_pilot "${KEYS[@]}" --skip-sdxl-frames 2>&1 | grep --line-buffered -E "done|lift points|Error|Traceback"
echo PASS1_DONE
rsync -a results_local/fill aurora:dev/single-view-nvs/results/ && rsync -a nvs/ aurora:dev/single-view-nvs/nvs/ && \
ssh aurora "cd ~/dev/single-view-nvs && . .venv/bin/activate && PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 nice python -m nvs.render_lifted ${KEYS[*]} 2>&1 | grep -E 'lifted|Error|Traceback'" && \
rsync -a aurora:dev/single-view-nvs/results/fill/ results_local/fill/ && echo LIFT_RENDERED
nice .venv-depth/bin/python -W ignore -m nvs.fill_pilot "${KEYS[@]}" 2>&1 | grep --line-buffered -E "done|Error|Traceback"
echo PASS2_DONE
