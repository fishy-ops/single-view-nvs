#!/bin/bash
# Data-scaling ablation: identical model and steps, 2k vs 7k training pairs.
cd ~/Documents/single-view-nvs
for N in 2000 0; do
  TAG=$([ "$N" = "0" ] && echo full || echo small)
  nice conda run --no-capture-output -n sharp python -W ignore -m nvs.train_bglayer --fold 0 --steps 4000 --crop 128 \
    --out bglayer_$TAG --max-train $N 2>&1 | grep --line-buffered -E "train pairs|step (2000|4000) |EVAL|skipped"
  echo TRAIN_${TAG}_DONE
  nice conda run --no-capture-output -n sharp python -W ignore -m nvs.eval_bglayer --ckpt results/bglayer_$TAG/fold0.pt 2>&1 | grep -E "FAIR|Traceback"
  echo EVAL_${TAG}_DONE
done
echo SCALING_DONE
