"""Naive "3D photo" baseline: one opaque isotropic Gaussian per source pixel, placed by a depth map.

--depth sharp    : depth = SHARP's own rendered source depth (isolates SHARP's learned Gaussian decoder)
--depth <method> : depth from results/depth/<method>/<scene>/srcNNN.npy (e.g. depthpro, computed on the Mac)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nvs.render_eval import DEV, ROOT, load_gaussians, render
from nvs.scenes import M360, SCENES, load_scene


def pixel_gaussians(img: np.ndarray, depth: torch.Tensor, K: np.ndarray, footprint: float = 0.8) -> dict:
    h, w = depth.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=DEV), torch.arange(w, device=DEV), indexing="ij")
    pix = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs)], -1).float()
    X = (pix @ torch.from_numpy(np.linalg.inv(K)).float().to(DEV).T) * depth[..., None]
    s = (depth / float(K[0, 0]) * footprint)[..., None].expand(h, w, 3)
    n = h * w
    return {
        "means": X.reshape(n, 3).half().cpu(),
        "quats": torch.tensor([1.0, 0, 0, 0]).expand(n, 4).half().clone(),
        "scales": s.reshape(n, 3).half().cpu(),
        "opacities": torch.ones(n).half(),
        "colors": torch.from_numpy(img.reshape(n, 3)).half() / 255,
        "color_space": "sRGB",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", default="sharp")
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    args = ap.parse_args()
    if args.scenes == ["m360"]:
        args.scenes = M360
    method = f"warp_{args.depth}depth" if args.depth == "sharp" else f"warp_{args.depth}"

    for name in args.scenes:
        scene = load_scene(name)
        for src_file in sorted((ROOT / "results" / "sharp" / name).glob("src*.pt")):
            si = int(src_file.stem[3:])
            sv = scene.views[si]
            img = np.asarray(Image.open(sv.path).convert("RGB"))
            if args.depth == "sharp":
                _, depth, _ = render(load_gaussians(src_file), np.eye(4), sv.K, scene.width, scene.height)
            else:
                depth = torch.from_numpy(np.load(ROOT / "results" / "depth" / args.depth / name / f"src{si:03d}.npy")).float().to(DEV)
            out = ROOT / "results" / method / name / src_file.name
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pixel_gaussians(img, depth, sv.K), out)
            print(method, name, src_file.name, flush=True)


if __name__ == "__main__":
    main()
