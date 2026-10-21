"""Generative-fill pilot, stage 1 (Mac): per-frame 2D fills + keyframe fills lifted to 3D points.

Input: results_local/paths/<key>/frames.npz (copied from the desktop's 1TB drive).
Outputs in results_local/fill/<key>/:
  lama.npz  : per-frame LaMa fill of the unseen mask
  sdxl.npz  : per-frame SDXL-inpainting fill (fixed seed; each frame independent, so it can flicker)
  lift.npz  : "fill once in 3D": SDXL fills of the two extreme frames, unseen pixels given
              background depth (max-depth dilation from visible neighbours) and unprojected into
              the source camera frame as points (xyz, rgb, radius). The desktop turns them into
              Gaussians, adds them to SHARP's and re-renders the path (nvs/render_lifted.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, grey_dilation, grey_erosion

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "results_local"
PROMPT = "a high quality photo, natural continuation of the scene"
NEG = "blurry, text, watermark, people, distorted"


def background_depth(depth, holes, iters=40):
    """Fill hole depth with the farthest valid neighbour: disoccluded surfaces sit behind the foreground."""
    d = np.where(holes, 0.0, depth.astype(np.float32))
    filled = ~holes
    for _ in range(iters):
        if filled.all():
            break
        grown = grey_dilation(d, size=(5, 5))
        new = ~filled & (grown > 0)
        d[new] = grown[new]
        filled |= new
    d[~filled] = d[filled].max() if filled.any() else 1.0
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--skip-sdxl-frames", action="store_true", help="only keyframe fills (for lift)")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lift-v2", action="store_true", help="4 keyframes, saved fills, edge-aware lifting -> lift_v2.npz")
    ap.add_argument("--keyframes", default="0,6,18,24")
    args = ap.parse_args()

    from diffusers import StableDiffusionXLInpaintPipeline
    from simple_lama_inpainting import SimpleLama

    dev = torch.device("mps")
    lama = SimpleLama(device=dev)
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
    ).to(dev)
    pipe.set_progress_bar_config(disable=True)

    def sdxl(rgb, mask, seed=0):
        h, w = rgb.shape[:2]
        H, W = int(round(h * 1024 / w / 8) * 8), 1024  # SDXL's native scale, then back down
        img = Image.fromarray(rgb).resize((W, H), Image.BICUBIC)
        m = Image.fromarray((mask * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
        g = torch.Generator("cpu").manual_seed(seed)
        out = pipe(prompt=PROMPT, negative_prompt=NEG, image=img, mask_image=m, height=H, width=W,
                   num_inference_steps=args.steps, guidance_scale=6.0, strength=0.99, generator=g).images[0]
        out = np.asarray(out.resize((w, h), Image.BICUBIC))
        return np.where(mask[..., None], out, rgb)  # keep seen pixels exactly

    for key in args.keys:
        z = np.load(LOCAL / "paths" / key / "frames.npz")
        rgb, unseen, depth, w2c, K = z["rgb"], z["unseen"], z["depth"], z["w2c"], z["K"]
        n = len(rgb)
        masks = np.stack([binary_dilation(u, iterations=2) for u in unseen])
        out = LOCAL / "fill" / key
        out.mkdir(parents=True, exist_ok=True)

        if not (out / "lama.npz").exists():
            lam = np.stack([np.asarray(lama(Image.fromarray(rgb[i]), Image.fromarray((masks[i] * 255).astype(np.uint8))))[: rgb.shape[1], : rgb.shape[2]] for i in range(n)])
            np.savez_compressed(out / "lama.npz", rgb=lam)
            print(key, "lama done", flush=True)

        if not args.skip_sdxl_frames and not (out / "sdxl.npz").exists():
            sd = np.stack([sdxl(rgb[i], masks[i], seed=0) for i in range(n)])
            np.savez_compressed(out / "sdxl.npz", rgb=sd)
            print(key, "sdxl per-frame done", flush=True)

        if args.lift_v2:
            kfs = [int(k) for k in args.keyframes.split(",") if int(k) < n]
            kf_path = out / "keyframes_v2.npz"
            if kf_path.exists():
                fills = dict(np.load(kf_path))
            else:
                fills = {f"f{i}": sdxl(rgb[i], masks[i], seed=1) for i in kfs}
                np.savez_compressed(kf_path, **fills)
            pts, cols, rads, kfi = [], [], [], []
            Kinv = np.linalg.inv(K)
            for i in kfs:
                holes = masks[i] | (z["alpha"][i] < 128)
                d = background_depth(depth[i], holes)
                vis_depth = np.where(holes, np.inf, depth[i].astype(np.float32))
                near_min = grey_erosion(vis_depth, size=(9, 9))  # nearest visible surface depth nearby
                silhouette = near_min < 0.9 * d  # a foreground object borders this hole pixel
                lift_mask = masks[i] & ~silhouette
                ys, xs = np.nonzero(lift_mask)
                pix = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs, dtype=np.float64)], 1)
                cam = (pix @ Kinv.T) * d[ys, xs, None]
                c2w = np.linalg.inv(w2c[i])
                pts.append(cam @ c2w[:3, :3].T + c2w[:3, 3])
                cols.append(fills[f"f{i}"][ys, xs])
                rads.append(d[ys, xs] / K[0, 0] * 0.8)
                kfi.append(np.full(len(ys), i, np.int16))
                print(key, "keyframe", i, "lifted", len(ys), "of", int(masks[i].sum()), "masked px", flush=True)
            np.savez_compressed(out / "lift_v2.npz", xyz=np.concatenate(pts).astype(np.float32), rgb=np.concatenate(cols).astype(np.uint8),
                                radius=np.concatenate(rads).astype(np.float32), kf=np.concatenate(kfi))
            continue

        if not (out / "lift.npz").exists():
            pts, cols, rads = [], [], []
            Kinv = np.linalg.inv(K)
            for i in [0, n - 1]:
                filled = sdxl(rgb[i], masks[i], seed=1)
                holes = masks[i]
                d = background_depth(depth[i], holes | (z["alpha"][i] < 128))
                ys, xs = np.nonzero(holes)
                pix = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs, dtype=np.float64)], 1)
                cam = (pix @ Kinv.T) * d[ys, xs, None]  # points in frame-i camera coordinates
                c2w = np.linalg.inv(w2c[i])  # frame camera -> source camera (the Gaussians' frame)
                world = cam @ c2w[:3, :3].T + c2w[:3, 3]
                pts.append(world)
                cols.append(filled[ys, xs])
                rads.append(d[ys, xs] / K[0, 0] * 0.8)
            np.savez_compressed(out / "lift.npz", xyz=np.concatenate(pts).astype(np.float32),
                                rgb=np.concatenate(cols).astype(np.uint8), radius=np.concatenate(rads).astype(np.float32))
            print(key, "lift points", sum(len(p) for p in pts), flush=True)


if __name__ == "__main__":
    main()
