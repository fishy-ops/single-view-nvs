"""Colour every unseen pixel of a camera sweep by what is hiding it (Mac, numpy only).

Same classification as nvs/unseen_anatomy.py, applied to the viewer sweeps: for each frame, take
SHARP's 3D point at each unseen pixel, push it back into the source camera (the sweep's middle
frame) and label it

  beyond frame  (blue)   : projects outside the source image
  behind object (orange) : sits well behind the source's visible surface there
  object surface (aqua)  : sits just behind it - the same surface curving out of view

Writes results_local/fill/<key>/anatomy.png (RGBA sprite, same grid as the other layers) and the
per-scene shares to results_local/fill/anatomy.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "results_local"
COLORS = {"beyond": (42, 120, 214), "behind": (235, 104, 52), "surface": (27, 175, 122)}


def classify(z, i, mid, thresh=0.05):
    H, W = z["unseen"][i].shape
    unseen = z["unseen"][i]
    d_t = z["depth"][i].astype(np.float32)
    d_s = z["depth"][mid].astype(np.float32)
    K = z["K"].astype(np.float64)
    T = z["w2c"][i].astype(np.float64)  # source camera -> this frame's camera
    med = float(np.median(d_s[d_s > 0]))
    ys, xs = np.nonzero(unseen)
    lab = np.zeros((H, W), np.uint8)  # 0 none, 1 beyond, 2 behind, 3 surface
    if len(ys) == 0:
        return lab, {"beyond": 0, "behind": 0, "surface": 0, "total": 0}
    pix = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs, dtype=np.float64)], 1)
    X_t = (pix @ np.linalg.inv(K).T) * d_t[ys, xs, None]
    R, t = T[:3, :3], T[:3, 3]
    X_s = (X_t - t) @ R
    z_s = X_s[:, 2]
    uv = X_s @ K.T
    u = uv[:, 0] / np.clip(z_s, 1e-6, None)
    v = uv[:, 1] / np.clip(z_s, 1e-6, None)
    inside = (z_s > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ui, vi = np.clip(u, 0, W - 1).astype(int), np.clip(v, 0, H - 1).astype(int)
    gap = (z_s - d_s[vi, ui]) / med
    lab[ys, xs] = np.where(~inside, 1, np.where(gap > thresh, 2, 3))
    counts = {"beyond": int((lab == 1).sum()), "behind": int((lab == 2).sum()),
              "surface": int((lab == 3).sum()), "total": int(unseen.sum())}
    return lab, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--width", type=int, default=448)
    args = ap.parse_args()
    out = {}
    for key in args.keys:
        z = np.load(LOCAL / "paths" / key / "frames.npz")
        n, H, W = z["rgb"].shape[:3]
        mid = n // 2
        cols = 5
        rows = (n + cols - 1) // cols
        Wt = args.width
        Ht = int(round(H * Wt / W))
        sheet = Image.new("RGBA", (Wt * cols, Ht * rows))
        agg = {"beyond": 0, "behind": 0, "surface": 0, "total": 0}
        for i in range(n):
            lab, c = classify(z, i, mid)
            for k in agg:
                agg[k] += c[k]
            rgba = np.zeros((H, W, 4), np.uint8)
            for idx, name in enumerate(["beyond", "behind", "surface"], start=1):
                m = lab == idx
                rgba[m] = (*COLORS[name], 165)
            tile = Image.fromarray(rgba).resize((Wt, Ht), Image.NEAREST)
            sheet.paste(tile, ((i % cols) * Wt, (i // cols) * Ht))
        sheet.save(LOCAL / "fill" / key / "anatomy.png", optimize=True)
        tot = max(agg["total"], 1)
        out[key] = {k: agg[k] / tot for k in ["beyond", "behind", "surface"]}
        print(f"{key:24s} beyond {100*out[key]['beyond']:5.1f}%  behind {100*out[key]['behind']:5.1f}%"
              f"  surface {100*out[key]['surface']:4.1f}%", flush=True)
    (LOCAL / "fill" / "anatomy.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
