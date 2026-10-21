"""Export per-pixel training tuples for the confidence-map experiment (desktop / CUDA).

For every (scene, source, target) pair used in the eval: the SHARP render at the target pose plus
test-time-available channels (alpha, relative depth, predicted-unseen mask), and the targets
(GT photo, LPIPS spatial map). Everything is resized to width 384. Output:
results/pairs/<scene>/s<src>_t<tgt>.npz
"""

from __future__ import annotations

import argparse
import json
import math

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nvs.render_eval import DEV, ROOT, covis_mask, load_gaussians, render
from nvs.scenes import M360, SCENES, load_scene, sparse_depth, target_indices

OUT_W = 384


def fit_scale(scene, sv, depth_s, alpha_s):
    uv, z = sparse_depth(scene, sv)
    ui = torch.from_numpy(uv[:, 0].astype(np.int64)).to(DEV)
    vi = torch.from_numpy(uv[:, 1].astype(np.int64)).to(DEV)
    d = depth_s[vi, ui].cpu().numpy()
    ok = (alpha_s[vi, ui].cpu().numpy() > 0.5) & (d > 1e-4)
    return float(np.median(z[ok] / d[ok])), float(np.median(z))


def down(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """HxWxC or HxW -> area-resized to h x w."""
    t = x if x.dim() == 3 else x[..., None]
    t = F.interpolate(t.permute(2, 0, 1)[None].float(), size=(h, w), mode="area")[0].permute(1, 2, 0)
    return t if x.dim() == 3 else t[..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=SCENES + M360)
    args = ap.parse_args()
    lp = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(DEV)
    n = 0
    for name in args.scenes:
        scene = load_scene(name)
        W, H = scene.width, scene.height
        h = int(round(H * OUT_W / W))
        out_dir = ROOT / "results" / "pairs" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for src_file in sorted((ROOT / "results" / "sharp" / name).glob("src*.pt")):
            si = int(src_file.stem[3:])
            sv = scene.views[si]
            g = load_gaussians(src_file)
            _, depth_s, alpha_s = render(g, np.eye(4), sv.K, W, H)
            scale, zmed = fit_scale(scene, sv, depth_s, alpha_s)
            dmed = float(depth_s[alpha_s > 0.5].median())
            for ti in target_indices(scene, si):
                path = out_dir / f"s{si:03d}_t{ti:03d}.npz"
                if path.exists():
                    continue
                tv = scene.views[ti]
                T = tv.w2c @ np.linalg.inv(sv.w2c)
                Tp = T.copy()
                Tp[:3, 3] /= scale
                with torch.no_grad():
                    rgb, depth_t, alpha_t = render(g, Tp, tv.K, W, H)
                    unseen = ~covis_mask(depth_t, alpha_t, depth_s, Tp, tv.K, sv.K, W, H)
                    gt = torch.from_numpy(np.asarray(Image.open(tv.path).convert("RGB"))).float().to(DEV) / 255
                    lmap = lp(rgb.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1)[0, 0]
                    err = (rgb - gt).abs().mean(-1)
                cos = (np.trace(T[:3, :3]) - 1) / 2
                np.savez_compressed(
                    path,
                    render=(down(rgb, h, OUT_W).clamp(0, 1) * 255).byte().cpu().numpy(),
                    gt=(down(gt, h, OUT_W) * 255).byte().cpu().numpy(),
                    alpha=(down(alpha_t, h, OUT_W).clamp(0, 1) * 255).byte().cpu().numpy(),
                    depth_rel=down(torch.log(depth_t.clamp(min=1e-4) / dmed), h, OUT_W).half().cpu().numpy(),
                    unseen=(down(unseen.float(), h, OUT_W) * 255).byte().cpu().numpy(),
                    lpips=down(lmap, h, OUT_W).half().cpu().numpy(),
                    abs_err=down(err, h, OUT_W).half().cpu().numpy(),
                    meta=json.dumps({"scene": name, "src": si, "tgt": ti,
                                     "baseline_rel": float(np.linalg.norm(T[:3, 3]) / zmed),
                                     "rot_deg": math.degrees(math.acos(max(-1.0, min(1.0, cos))))}),
                )
                n += 1
        print("pairs", name, n, flush=True)


if __name__ == "__main__":
    main()
