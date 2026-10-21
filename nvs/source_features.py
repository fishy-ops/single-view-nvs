"""Per-source features computable from the single input image + SHARP output only (no GT views).

Written to results/source_features.jsonl; joined with eval rows by nvs/predictability.py.
"""

from __future__ import annotations

import argparse
import json

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nvs.render_eval import DEV, ROOT, load_gaussians, psnr, render
from nvs.scenes import M360, SCENES, load_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=SCENES + M360)
    args = ap.parse_args()
    lp = lpips.LPIPS(net="alex", verbose=False).to(DEV)
    out = ROOT / "results" / "source_features.jsonl"
    done = set()
    if out.exists():
        done = {(r["scene"], r["src"]) for r in map(json.loads, out.read_text().splitlines())}
    with open(out, "a") as f:
        for name in args.scenes:
            files = sorted((ROOT / "results" / "sharp" / name).glob("src*.pt"))
            if not files:
                continue
            scene = load_scene(name)
            for src_file in files:
                si = int(src_file.stem[3:])
                if (name, si) in done:
                    continue
                g = load_gaussians(src_file)
                sv = scene.views[si]
                img = torch.from_numpy(np.asarray(Image.open(sv.path).convert("RGB"))).float().to(DEV) / 255
                rgb, depth, alpha = render(g, np.eye(4), sv.K, scene.width, scene.height)

                ld = torch.log(depth.clamp(min=1e-4))
                gx, gy = ld[:, 1:] - ld[:, :-1], ld[1:] - ld[:-1]
                gray = img.mean(-1)
                ix, iy = gray[:, 1:] - gray[:, :-1], gray[1:] - gray[:-1]
                op = g["opacities"]
                d = depth[alpha > 0.5]
                row = {
                    "scene": name,
                    "src": si,
                    "self_psnr": psnr(rgb, img),
                    "self_lpips": float(lp(rgb.permute(2, 0, 1)[None] * 2 - 1, img.permute(2, 0, 1)[None] * 2 - 1)),
                    "depth_edge_frac": float(((gx.abs() > 0.05).float().mean() + (gy.abs() > 0.05).float().mean()) / 2),
                    "depth_log_range": float(torch.quantile(torch.log(d[::17]), 0.95) - torch.quantile(torch.log(d[::17]), 0.05)),
                    "near_frac": float((d < 0.5 * d.median()).float().mean()),
                    "texture": float((ix.abs().mean() + iy.abs().mean()) / 2),
                    "opacity_mean": float(op.mean()),
                    "opacity_lowfrac": float((op < 0.1).float().mean()),
                    "scale_median": float(g["scales"].max(-1).values.median() / depth.median()),
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()
