"""Pre-render a camera grid for the web demo (desktop / CUDA).

For a chosen source view, cameras sit on a 7x7 grid of offsets in the source image plane, up to
+-rmax x median predicted depth, each looking at the scene centre (0, 0, D). Outputs per scene in
results/viewer/<scene>_src<N>/:
  sharp.jpg, warp.jpg : sprite sheets (row = dy index, col = dx index)
  mask.png            : sprite of "not seen by the source photo" masks (red, alpha)
  source.jpg          : the input photo
  cells.json          : per cell offset, baseline_rel, rot_deg, disocc_pred, hole_frac
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
from PIL import Image

from nvs.make_warp import pixel_gaussians
from nvs.render_eval import DEV, ROOT, covis_mask, load_gaussians, render
from nvs.scenes import load_scene


def look_at_w2c(c: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = target - c
    z /= np.linalg.norm(z)
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)  # OpenCV: y down, so up-vector hint is +y
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ c
    return w2c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+", help="scene:src, e.g. fortress:21")
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--rmax", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=480)
    args = ap.parse_args()

    for item in args.items:
        name, si = item.split(":")
        si = int(si)
        scene = load_scene(name)
        sv = scene.views[si]
        g = load_gaussians(ROOT / "results" / "sharp" / name / f"src{si:03d}.pt")
        W0, H0 = scene.width, scene.height
        s = args.width / W0
        W, H = args.width, int(round(H0 * s))
        K = sv.K.copy()
        K[:2] *= s

        _, depth_full, _ = render(g, np.eye(4), sv.K, W0, H0)
        img_full = np.asarray(Image.open(sv.path).convert("RGB"))
        warp = pixel_gaussians(img_full, depth_full, sv.K)
        warp = {k: (v.to(DEV).float() if torch.is_tensor(v) else v) for k, v in warp.items()}
        _, depth_s, _ = render(g, np.eye(4), K, W, H)
        D = float(depth_s.median())

        n = args.grid
        sheets = {k: np.zeros((H * n, W * n, 3 if k != "mask" else 4), np.uint8) for k in ["sharp", "warp", "mask"]}
        cells = []
        conf_in = np.zeros((n, n, 3, H, W), np.float16)  # alpha, depth_rel, unseen, same encoding as export_pairs
        dmed = float(depth_s[depth_s > 0].median())
        offs = np.linspace(-1, 1, n) * args.rmax
        for iy, oy in enumerate(offs):
            for ix, ox in enumerate(offs):
                c = np.array([ox * D, oy * D, 0.0])
                w2c = look_at_w2c(c, np.array([0.0, 0.0, D]))
                rgb, depth_t, alpha_t = render(g, w2c, K, W, H)
                wrgb, _, _ = render(warp, w2c, K, W, H)
                mask = covis_mask(depth_t, alpha_t, depth_s, w2c, K, K, W, H)
                sl = np.s_[iy * H:(iy + 1) * H, ix * W:(ix + 1) * W]
                conf_in[iy, ix, 0] = alpha_t.clamp(0, 1).cpu().numpy()
                conf_in[iy, ix, 1] = (torch.log(depth_t.clamp(min=1e-4) / dmed).clamp(-3, 3) / 3).cpu().numpy()
                conf_in[iy, ix, 2] = (~mask).float().cpu().numpy()
                sheets["sharp"][sl] = (rgb.cpu().numpy() * 255).astype(np.uint8)
                sheets["warp"][sl] = (wrgb.cpu().numpy() * 255).astype(np.uint8)
                m = (~mask).cpu().numpy()
                sheets["mask"][sl] = np.where(m[..., None], [235, 64, 52, 150], [0, 0, 0, 0]).astype(np.uint8)
                cos = (np.trace(w2c[:3, :3]) - 1) / 2
                cells.append({"ix": ix, "iy": iy, "ox": float(ox), "oy": float(oy),
                              "baseline_rel": float(np.linalg.norm(c) / D),
                              "rot_deg": math.degrees(math.acos(max(-1, min(1, cos)))),
                              "disocc_pred": float(1 - mask.float().mean()),
                              "hole_frac": float(1 - alpha_t.mean())})
        out = ROOT / "results" / "viewer" / f"{name}_src{si:03d}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "conf_inputs.npy", conf_in)
        Image.fromarray(sheets["sharp"]).save(out / "sharp.jpg", quality=82, optimize=True)
        Image.fromarray(sheets["warp"]).save(out / "warp.jpg", quality=82, optimize=True)
        Image.fromarray(sheets["mask"]).save(out / "mask.png", optimize=True)
        Image.fromarray(img_full).resize((W, H), Image.LANCZOS).save(out / "source.jpg", quality=88)
        (out / "cells.json").write_text(json.dumps({"scene": name, "src": si, "grid": n, "w": W, "h": H,
                                                    "rmax": args.rmax, "cells": cells}))
        print("viewer", item, W, H, flush=True)


if __name__ == "__main__":
    main()
