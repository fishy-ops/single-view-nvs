"""Readable confidence-map figures from the saved fold models (Mac).

Strip per scene: SHARP render | real photo | measured LPIPS map | predicted map. Both maps share
one scale (2nd-98th percentile of the measured map) on a single-hue light-to-dark ramp. Uses the
median-baseline test pair, so the example shows a typical move rather than the extreme.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from PIL import Image

from nvs.train_confidence import ROOT, UNet, folds, load_pairs

LO, HI = np.array([250, 240, 236]), np.array([122, 26, 16])


def ramp(m, lo, hi):
    t = np.clip((m - lo) / (hi - lo + 1e-6), 0, 1)[..., None]
    return (LO * (1 - t) + HI * t).astype(np.uint8)


def main():
    run, scenes = sys.argv[1], sys.argv[2:]
    OUT = ROOT / "results" / run
    X, Y, meta = load_pairs()
    fold_of = folds(meta)
    dev = torch.device("mps")
    models = {}
    for s in scenes:
        k = fold_of[s]
        if k not in models:
            m = UNet().to(dev).eval()
            m.load_state_dict(torch.load(OUT / f"unet_fold{k}.pt", map_location=dev))
            models[k] = m
        idx = sorted((i for i, mm in enumerate(meta) if mm["scene"] == s), key=lambda i: meta[i]["baseline_rel"])
        i = idx[len(idx) // 2]
        with torch.no_grad():
            pred = models[k](X[i][None].float().to(dev))[0, 0].cpu().numpy()
        y = Y[i][0].float().numpy()
        lo, hi = np.percentile(y, 2), np.percentile(y, 98)
        z = np.load(ROOT / "results/pairs" / s / f"s{meta[i]['src']:03d}_t{meta[i]['tgt']:03d}.npz")
        strip = np.concatenate([z["render"], z["gt"], ramp(y, lo, hi), ramp(pred, lo, hi)], 1)
        Image.fromarray(strip).save(OUT / f"fig_{s}.jpg", quality=88)
        print(s, "fold", k, "baseline", round(meta[i]["baseline_rel"], 3))


if __name__ == "__main__":
    main()
