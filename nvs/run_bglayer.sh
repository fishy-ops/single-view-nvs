#!/bin/bash
# Overnight: wait for the desktop export, sync pairs to the Mac, train the background-layer head.
cd ~/Documents/single-view-nvs
EXPECT=$(ls results/pairs/*/ | grep -c npz)   # same pair set as the confidence experiment
echo "expecting ~$EXPECT training pairs"
for i in $(seq 1 60); do
  N=$(ssh -o ConnectTimeout=10 aurora 'ls ~/dev/single-view-nvs/results/train_pairs/*/ 2>/dev/null | grep -c npz' || echo 0)
  [ "$N" -ge "$EXPECT" ] && break
  sleep 60
done
echo "desktop has $N pairs; syncing"
for i in 1 2 3 4 5; do
  rsync -a --partial --timeout=60 aurora:dev/single-view-nvs/results/train_pairs results_local/ && break
  echo "rsync retry $i"; sleep 15
done
M=$(ls results_local/train_pairs/*/ | grep -c npz)
echo "mac has $M pairs"
[ "$M" -lt 1000 ] && { echo "TOO FEW PAIRS, abort"; exit 1; }
nice conda run --no-capture-output -n sharp python -W ignore -m nvs.train_bglayer --fold 0 --steps 6000 2>&1 | grep --line-buffered -vE "Warning|warn"
echo BGLAYER_DONE
