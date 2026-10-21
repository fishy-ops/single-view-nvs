"""Temporal stability of fills along a camera path (Mac).

Optical flow (torchvision RAFT-large) is computed once on the *raw SHARP* frames, so every method is
warped with the same motion. For consecutive frames t, t+1 we backward-warp method output t+1 into
frame t and measure mean absolute colour change (0-255) over pixels that are unseen in frame t and
pass a forward-backward flow check. Lower = the fill holds still as the camera moves. The same
number over seen pixels is reported as a sanity floor (should be similar across methods).
Output: results_local/fill/stability.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "results_local"
METHODS = ["raw", "lama", "sdxl", "lifted", "layered", "lifted_v2", "layered_v2", "bglayer"]


def warp(img, flow):
    """Backward warp img (B,C,H,W) with flow (B,2,H,W) mapping target pixel -> source pixel."""
    b, _, h, w = img.shape
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    grid = torch.stack([xs, ys], 0)[None].float() + flow
    grid = torch.stack([grid[:, 0] / (w - 1) * 2 - 1, grid[:, 1] / (h - 1) * 2 - 1], -1)
    return F.grid_sample(img, grid, align_corners=True, mode="bilinear", padding_mode="border")


def main():
    keys = sys.argv[1:]
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).eval()
    tf = weights.transforms()
    results = {}
    for key in keys:
        z = np.load(LOCAL / "paths" / key / "frames.npz")
        raw, unseen = z["rgb"], z["unseen"]
        n, H, W = raw.shape[:3]
        H8, W8 = H // 8 * 8, W // 8 * 8
        outs = {"raw": raw}
        for m in METHODS[1:]:
            f = LOCAL / "fill" / key / f"{m}.npz"
            if f.exists():
                outs[m] = np.load(f)["rgb"]
        to_t = lambda a: torch.from_numpy(a[:H8, :W8]).permute(2, 0, 1)[None].float()
        per = {m: {"unseen": [], "seen": []} for m in outs}
        for t in range(n - 1):
            a, b = to_t(raw[t]) / 255, to_t(raw[t + 1]) / 255
            with torch.no_grad():
                i1, i2 = tf(a, b)
                fwd = model(i1, i2)[-1]  # t -> t+1
                bwd = model(i2, i1)[-1]  # t+1 -> t
            back = warp(bwd, fwd)  # bwd sampled at forward-displaced positions
            ok = ((fwd + back).norm(dim=1) < 1.5)[0].numpy()
            u = unseen[t][:H8, :W8] & ok
            s = ~unseen[t][:H8, :W8] & ok
            for m, frames in outs.items():
                cur = to_t(frames[t])
                nxt_w = warp(to_t(frames[t + 1]), fwd)
                diff = (cur - nxt_w).abs().mean(1)[0].numpy()
                if u.sum() > 50:
                    per[m]["unseen"].append(float(diff[u].mean()))
                per[m]["seen"].append(float(diff[s].mean()))
        results[key] = {m: {k: float(np.mean(v)) if v else None for k, v in d.items()} for m, d in per.items()}
        print(key, {m: round(v["unseen"], 2) if v["unseen"] else None for m, v in results[key].items()}, flush=True)
    summary = {m: float(np.mean([results[k][m]["unseen"] for k in results if m in results[k] and results[k][m]["unseen"] is not None]))
               for m in METHODS if any(m in results[k] for k in results)}
    results["mean_unseen_change"] = summary
    (LOCAL / "fill" / "stability.json").write_text(json.dumps(results, indent=1))
    print("MEAN unseen frame-to-frame change (0-255):", {k: round(v, 2) for k, v in summary.items()})


if __name__ == "__main__":
    main()
