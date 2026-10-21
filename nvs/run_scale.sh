#!/bin/bash
# Scale-up chain: wait for SHARP 30-source predictions -> export pairs (desktop) -> sync -> retrain v2.
cd ~/Documents/single-view-nvs
while pgrep -f "[n]vs.predict_sharp" >/dev/null; do sleep 60; done
echo PREDICT_DONE $(ls results/sharp/*/ | grep -c pt) sources
rsync -a results/sharp aurora:dev/single-view-nvs/results/ || exit 1
ssh aurora "cd ~/dev/single-view-nvs && . .venv/bin/activate && PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 nice python -m nvs.export_train_pairs 2>&1 | grep -E 'train_pairs|Error|Traceback|No space'"
echo EXPORT_DONE
for i in 1 2 3 4 5; do
  rsync -a --partial --timeout=60 aurora:dev/single-view-nvs/results/train_pairs results_local/ && break
  echo "rsync retry $i"; sleep 20
done
echo "mac pairs: $(ls results_local/train_pairs/*/ | grep -c npz)"
df -h ~ | tail -1
nice conda run --no-capture-output -n sharp python -W ignore -m nvs.train_bglayer --fold 0 --steps 9000 --crop 160 --out bglayer_v2 2>&1 | grep --line-buffered -vE "Warning|warn"
echo TRAIN_V2_DONE
nice conda run --no-capture-output -n sharp python -W ignore -m nvs.eval_bglayer --ckpt results/bglayer_v2/fold0.pt 2>&1 | grep -E "FAIR|Error|Traceback"
echo EVAL_V2_DONE
