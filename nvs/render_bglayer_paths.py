"""Render the viewer camera sweeps with the trained background layer (Mac / MPS).

The middle frame of each sweep is the source view itself (zero offset, identity pose), so its RGB,
depth and alpha are the head's input. The predicted layer is unprojected once into the source
camera frame and rendered at every frame with nvs/splat_torch.py, then shown only inside that
frame's unseen mask, exactly like the "layered" baselines. Output:
results_local/fill/<key>/bglayer.npz
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from nvs.splat_torch import render_torch
from nvs.train_bglayer import BGNet, ROOT, background_estimate, features, layer_to_points

LOCAL = ROOT / "results_local"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--ckpt", default="results/bglayer/fold0.pt")
    ap.add_argument("--width", type=int, default=384)
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = BGNet().to(dev).eval()
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=dev))

    for key in args.keys:
        z = np.load(LOCAL / "paths" / key / "frames.npz")
        n, H0, W0 = z["rgb"].shape[:3]
        s = args.width / W0
        W, H = args.width, int(round(H0 * s)) // 8 * 8
        K = z["K"].astype(np.float32).copy()
        K[:2] *= s
        mid = n // 2  # zero-offset frame == the source view
        rs = lambda a, mode="bilinear": F.interpolate(torch.from_numpy(np.ascontiguousarray(a)).float()[None, None], size=(H, W), mode=mode)[0, 0]
        p = {
            "src_rgb": F.interpolate(torch.from_numpy(z["rgb"][mid]).float().permute(2, 0, 1)[None] / 255, size=(H, W), mode="bilinear")[0],
            "src_depth": rs(z["depth"][mid].astype(np.float32)),
            "src_alpha": rs(z["alpha"][mid].astype(np.float32) / 255),
            "K": torch.from_numpy(K), "hw": (H, W),
        }
        p["bg_init"] = background_estimate(p)
        with torch.no_grad():
            out = model(features(p)[None].to(dev))[0]
            xyz, quats, radius, op, col = layer_to_points(out, p, dev)
            frames = []
            for i in range(n):
                w2c = torch.from_numpy(z["w2c"][i].astype(np.float32)).to(dev)
                Kt = torch.from_numpy(K).to(dev)
                rgb, _, alpha = render_torch(xyz, quats, radius, op, col, w2c, Kt, W, H)
                rgb = F.interpolate(rgb.permute(2, 0, 1)[None], size=(H0, W0), mode="bilinear")[0].permute(1, 2, 0)
                al = F.interpolate(alpha[None, None], size=(H0, W0), mode="bilinear")[0, 0]
                unseen = cv2.dilate(z["unseen"][i].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                use = unseen & (al.cpu().numpy() > 0.5)
                img = np.where(use[..., None], (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), z["rgb"][i])
                left = (unseen & ~use).astype(np.uint8)
                frames.append(cv2.inpaint(img, left, 5, cv2.INPAINT_TELEA) if left.any() else img)
        np.savez_compressed(LOCAL / "fill" / key / "bglayer.npz", rgb=np.stack(frames))
        print("bglayer", key, "frames", len(frames), flush=True)


if __name__ == "__main__":
    main()
