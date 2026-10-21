"""Beyond-the-frame layer: outpaint the source photo, lift it to 3D, render along the sweep (Mac).

25% of the unseen pixels in these sweeps lie outside the source frame, where a layer built from
source pixels cannot reach. Here the source view is placed on a canvas 1.5x wider, SDXL outpaints
the border, the new pixels get depth extrapolated from the frame edge, and they become 3D points in
the source camera frame - so, like the other 3D fills, they cannot flicker.

Output: results_local/fill/<key>/outpaint.npz (path frames, composited inside the unseen mask).
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nvs.splat_torch import render_torch
from nvs.train_bglayer import ROOT

LOCAL = ROOT / "results_local"
PROMPT = "a high quality photo, natural continuation of the scene"
NEG = "blurry, text, watermark, frame, border, distorted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--expand", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()
    dev = torch.device("mps")
    from diffusers import StableDiffusionXLInpaintPipeline

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to(dev)
    pipe.set_progress_bar_config(disable=True)

    for key in args.keys:
        z = np.load(LOCAL / "paths" / key / "frames.npz")
        n, H, W = z["rgb"].shape[:3]
        mid = n // 2
        K = z["K"].astype(np.float64).copy()
        W2, H2 = int(W * args.expand) // 8 * 8, int(H * args.expand) // 8 * 8
        ox, oy = (W2 - W) // 2, (H2 - H) // 2
        K2 = K.copy()
        K2[0, 2] += ox
        K2[1, 2] += oy  # same focal, shifted principal point: a wider view of the same camera

        canvas = np.zeros((H2, W2, 3), np.uint8)
        canvas[oy:oy + H, ox:ox + W] = z["rgb"][mid]
        mask = np.ones((H2, W2), np.uint8)
        mask[oy:oy + H, ox:ox + W] = 0

        img = Image.fromarray(canvas).resize((1024, int(H2 * 1024 / W2) // 8 * 8), Image.BICUBIC)
        m = Image.fromarray(mask * 255).resize(img.size, Image.NEAREST)
        out = pipe(prompt=PROMPT, negative_prompt=NEG, image=img, mask_image=m, height=img.size[1], width=img.size[0],
                   num_inference_steps=args.steps, guidance_scale=6.0, strength=0.99,
                   generator=torch.Generator("cpu").manual_seed(0)).images[0]
        filled = np.asarray(out.resize((W2, H2), Image.BICUBIC)).copy()
        filled[oy:oy + H, ox:ox + W] = z["rgb"][mid]  # keep the real photo exactly

        # Depth for the new border: replicate the nearest frame-edge depth outward.
        d_src = z["depth"][mid].astype(np.float32)
        d2 = np.zeros((H2, W2), np.float32)
        d2[oy:oy + H, ox:ox + W] = d_src
        d2 = cv2.inpaint((d2 / max(d2.max(), 1e-6) * 255).astype(np.uint8), mask, 9, cv2.INPAINT_NS).astype(np.float32) / 255 * max(d2.max(), 1e-6)
        d2[oy:oy + H, ox:ox + W] = d_src

        ys, xs = np.nonzero(mask)
        pix = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs, dtype=np.float64)], 1)
        xyz = (pix @ np.linalg.inv(K2).T) * d2[ys, xs, None]
        xyz_t = torch.from_numpy(xyz).float().to(dev)
        col = torch.from_numpy(filled[ys, xs].astype(np.float32) / 255).to(dev)
        rad = torch.from_numpy((d2[ys, xs] / K2[0, 0] * 1.1).astype(np.float32)).to(dev)[:, None].expand(-1, 3)
        quats = torch.tensor([1.0, 0, 0, 0], device=dev).expand(len(xyz), 4)
        op = torch.ones(len(xyz), device=dev)
        print(key, "outpainted border points", len(xyz), flush=True)

        frames = []
        kernel = np.ones((5, 5), np.uint8)
        with torch.no_grad():
            for i in range(n):
                w2c = torch.from_numpy(z["w2c"][i].astype(np.float32)).to(dev)
                Kt = torch.from_numpy(K.astype(np.float32)).to(dev)
                rgb, _, alpha = render_torch(xyz_t, quats, rad, op, col, w2c, Kt, W, H)
                unseen = cv2.dilate(z["unseen"][i].astype(np.uint8), kernel) > 0
                use = unseen & (alpha.cpu().numpy() > 0.5)
                img_i = np.where(use[..., None], (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), z["rgb"][i])
                left = (unseen & ~use).astype(np.uint8)
                frames.append(cv2.inpaint(img_i, left, 5, cv2.INPAINT_TELEA) if left.any() else img_i)
        np.savez_compressed(LOCAL / "fill" / key / "outpaint.npz", rgb=np.stack(frames))
        print(key, "outpaint frames", len(frames), flush=True)


if __name__ == "__main__":
    main()
