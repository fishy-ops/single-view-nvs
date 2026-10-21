"""What is actually hidden? Split the unseen region into three kinds (Mac, numpy only).

For every unseen pixel in a target view, take SHARP's 3D point there, push it back into the source
camera, and ask what stood in the way:

  beyond_frame  : it projects outside the source image - no pixel could ever have seen it.
  behind_object : it sits well behind the source's visible surface at that pixel - background that a
                  foreground object was covering.
  near_surface  : it sits only just behind that surface - the same object's own side curving out of
                  view, or a grazing silhouette.

"Just behind" is a depth gap under `--thresh` x the scene's median depth (default 5%).

Caveat: the 3D point comes from SHARP's rendered depth in a region SHARP never saw, so the split is
as good as that depth. The beyond_frame share needs no depth and is exact.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "results_local" / "train_pairs"


def anatomy(z, thresh):
    unseen = z["unseen"]
    if unseen.sum() < 100:
        return None
    H, W = unseen.shape
    d_t = z["tgt_depth"].astype(np.float32)
    d_s = z["src_depth"].astype(np.float32)
    K, Kt, T = z["K"].astype(np.float64), z["K_tgt"].astype(np.float64), z["w2c"].astype(np.float64)
    med = float(np.median(d_s[d_s > 0]))

    ys, xs = np.nonzero(unseen)
    pix = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs, dtype=np.float64)], 1)
    X_t = (pix @ np.linalg.inv(Kt).T) * d_t[ys, xs, None]
    R, t = T[:3, :3], T[:3, 3]
    X_s = (X_t - t) @ R  # inverse rotation: R^T (X_t - t)
    z_s = X_s[:, 2]
    uv = (X_s @ K.T)
    u = uv[:, 0] / np.clip(z_s, 1e-6, None)
    v = uv[:, 1] / np.clip(z_s, 1e-6, None)

    inside = (z_s > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out = {"beyond_frame": int((~inside).sum())}
    ui = np.clip(u, 0, W - 1).astype(int)
    vi = np.clip(v, 0, H - 1).astype(int)
    gap = (z_s - d_s[vi, ui]) / med
    hidden = inside & (z_s > 0)
    out["behind_object"] = int((hidden & (gap > thresh)).sum())
    out["near_surface"] = int((hidden & (gap <= thresh)).sum())
    out["total"] = int(unseen.sum())
    out["unseen_frac"] = float(unseen.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=0.05, help="depth gap as a fraction of scene depth")
    ap.add_argument("--per-scene", type=int, default=40)
    args = ap.parse_args()

    by_scene = defaultdict(list)
    for f in sorted(PAIRS.glob("*/*.npz")):
        by_scene[f.parent.name].append(f)

    agg, rows = defaultdict(int), []
    for scene, files in sorted(by_scene.items()):
        idx = np.linspace(0, len(files) - 1, min(args.per_scene, len(files))).round().astype(int)
        s_agg = defaultdict(int)
        for i in sorted(set(idx)):
            a = anatomy(np.load(files[i]), args.thresh)
            if a is None:
                continue
            for k in ["beyond_frame", "behind_object", "near_surface", "total"]:
                s_agg[k] += a[k]
                agg[k] += a[k]
        if s_agg["total"]:
            rows.append({"scene": scene, **{k: s_agg[k] / s_agg["total"] for k in ["beyond_frame", "behind_object", "near_surface"]},
                         "pixels": s_agg["total"]})
    print(f"{'scene':16s} {'beyond frame':>13s} {'behind object':>14s} {'object surface':>15s}")
    for r in rows:
        print(f"{r['scene']:16s} {100*r['beyond_frame']:12.1f}% {100*r['behind_object']:13.1f}% {100*r['near_surface']:14.1f}%")
    tot = agg["total"]
    print(f"{'ALL':16s} {100*agg['beyond_frame']/tot:12.1f}% {100*agg['behind_object']/tot:13.1f}% {100*agg['near_surface']/tot:14.1f}%")
    (ROOT / "results" / "unseen_anatomy.json").write_text(json.dumps(
        {"threshold_frac_of_scene_depth": args.thresh, "per_scene": rows,
         "all": {k: agg[k] / tot for k in ["beyond_frame", "behind_object", "near_surface"]}, "pixels": tot}, indent=1))


if __name__ == "__main__":
    main()
