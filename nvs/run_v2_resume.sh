#!/bin/bash
# Resume v2 training (smaller crops, lighter perceptual term), then eval, paths, stability, viewer data.
cd ~/Documents/single-view-nvs
nice conda run --no-capture-output -n sharp python -W ignore -m nvs.train_bglayer --fold 0 --steps 4000 --crop 128 --out bglayer_v2 --resume 2>&1 | grep --line-buffered -vE "Warning|warn"
echo TRAIN_V2_DONE
nice conda run --no-capture-output -n sharp python -W ignore -m nvs.eval_bglayer --ckpt results/bglayer_v2/fold0.pt 2>&1 | grep -E "FAIR|Error|Traceback"
echo EVAL_V2_DONE
