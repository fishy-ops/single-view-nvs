#!/bin/bash
# Resume background-layer training with crop rendering (lower memory), then render the viewer paths.
cd ~/Documents/single-view-nvs
nice conda run --no-capture-output -n sharp python -W ignore -m nvs.train_bglayer --fold 0 --steps 6000 --crop 160 --resume 2>&1 | grep --line-buffered -vE "Warning|warn"
echo BGLAYER_DONE
nice conda run --no-capture-output -n sharp python -W ignore -m nvs.render_bglayer_paths m360_kitchen_src030 fortress_src021 m360_counter_src117 horns_src033 m360_bonsai_src012 2>&1 | grep --line-buffered -E "bglayer|Error|Traceback"
echo PATHS_DONE
nice .venv-depth/bin/python -W ignore -m nvs.fill_eval m360_kitchen_src030 fortress_src021 m360_counter_src117 horns_src033 m360_bonsai_src012 2>&1 | grep -vE "Warning|warn" | tail -8
echo STABILITY_DONE
