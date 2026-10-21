"""Predicted-error sprite for the web viewer (Mac): runs the confidence UNet on every grid cell.

Each scene uses the fold model for which that scene was held out, so the demo shows
out-of-scene predictions. Output: results/viewer/<key>/conf.png (RGBA sprite, alpha = predicted
LPIPS on a fixed scale shared by all scenes).
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from PIL import Image

from nvs.train_confidence import ROOT, UNet, folds
import json

LO, HI = 0.35, 0.8  # below ~median LPIPS stays clear


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "confidence"
    keys = sys.argv[2:]
    scenes = sorted({p.parent.name for p in (ROOT / "results/pairs").glob("*/*.npz")})
    fold_of = folds([{"scene": s} for s in scenes])
    dev = torch.device("mps")
    for key in keys:
        d = ROOT / "results/viewer" / key
        cells = json.loads((d / "cells.json").read_text())
        n, W, H = cells["grid"], cells["w"], cells["h"]
        model = UNet().to(dev).eval()
        model.load_state_dict(torch.load(ROOT / "results" / run / f"unet_fold{fold_of[cells['scene']]}.pt", map_location=dev))
        rgb = np.asarray(Image.open(d / "sharp.jpg").convert("RGB"), np.float32) / 255
        aux = np.load(d / "conf_inputs.npy").astype(np.float32)
        sheet = np.zeros((H * n, W * n, 4), np.uint8)
        for iy in range(n):
            for ix in range(n):
                tile = rgb[iy * H:(iy + 1) * H, ix * W:(ix + 1) * W].transpose(2, 0, 1)
                x = torch.from_numpy(np.concatenate([tile, aux[iy, ix]], 0))[None].to(dev)
                with torch.no_grad():
                    p = model(x)[0, 0].cpu().numpy()
                a = np.clip((p - LO) / (HI - LO), 0, 1)
                sheet[iy * H:(iy + 1) * H, ix * W:(ix + 1) * W] = np.stack(
                    [np.full_like(a, 200), np.full_like(a, 30), np.full_like(a, 20), a * 215], -1).astype(np.uint8)
        Image.fromarray(sheet).save(d / "conf.png", optimize=True)
        print(key, "fold", fold_of[cells["scene"]], "conf.png", (d / "conf.png").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
