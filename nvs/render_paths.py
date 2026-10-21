"""Smooth camera paths for the generative-fill pilot (desktop / CUDA).

For each scene:source, render a horizontal-then-vertical sweep that looks at the scene centre:
N frames with offsets from -rmax to +rmax x median predicted depth. Per frame: SHARP RGB,
alpha, depth (source-camera units), unseen mask, and the camera. Written to
results/paths/<scene>_src<N>/frames.npz (results/ is a symlink onto the 1TB HDD).
Temporal-consistency metrics need consecutive frames with known geometry, which the
7x7 viewer grid is too coarse to give.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from nvs.export_viewer import look_at_w2c
from nvs.render_eval import ROOT, covis_mask, load_gaussians, render
from nvs.scenes import load_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+", help="scene:src")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--rmax", type=float, default=0.3)
    ap.add_argument("--width", type=int, default=512)
    args = ap.parse_args()

    for item in args.items:
        name, si = item.split(":")
        si = int(si)
        scene = load_scene(name)
        sv = scene.views[si]
        g = load_gaussians(ROOT / "results" / "sharp" / name / f"src{si:03d}.pt")
        s = args.width / scene.width
        W, H = args.width, int(round(scene.height * s)) // 8 * 8  # SDXL wants multiples of 8
        K = sv.K.copy()
        K[:2] *= s
        _, depth_s, _ = render(g, np.eye(4), K, W, H)
        D = float(depth_s.median())
        offs = np.linspace(-1, 1, args.frames) * args.rmax
        cams = [np.array([o * D, 0.0, 0.0]) for o in offs]  # horizontal sweep (the common parallax gesture)
        out = {k: [] for k in ["rgb", "alpha", "depth", "unseen", "w2c"]}
        with torch.no_grad():
            for c in cams:
                w2c = look_at_w2c(c, np.array([0.0, 0.0, D]))
                rgb, depth_t, alpha_t = render(g, w2c, K, W, H)
                unseen = ~covis_mask(depth_t, alpha_t, depth_s, w2c, K, K, W, H)
                out["rgb"].append((rgb.cpu().numpy() * 255).astype(np.uint8))
                out["alpha"].append((alpha_t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
                out["depth"].append(depth_t.half().cpu().numpy())
                out["unseen"].append(unseen.cpu().numpy())
                out["w2c"].append(w2c.astype(np.float32))
        d = ROOT / "results" / "paths" / f"{name}_src{si:03d}"
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "frames.npz", K=K.astype(np.float32), D=D, offsets=offs,
                            **{k: np.stack(v) for k, v in out.items()})
        print("path", item, W, H, "unseen share per frame", np.round(np.stack(out["unseen"]).mean((1, 2))[:: max(1, args.frames // 5)], 3), flush=True)


if __name__ == "__main__":
    main()
