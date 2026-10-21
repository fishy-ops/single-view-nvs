"""Re-render a camera path with SHARP Gaussians + lifted fill points (desktop / CUDA).

Input: results/paths/<key>/frames.npz (cameras) and results/fill/<key>/lift.npz (points from
nvs/fill_pilot.py, in the source-camera frame, sRGB colours). Fill points become small isotropic
opaque Gaussians. Colours are converted to linear RGB to match SHARP. Any remaining alpha holes get
a Telea fill. Output: results/fill/<key>/lifted.npz (rgb frames).
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import torch

from nvs.render_eval import DEV, ROOT, load_gaussians, render


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def main():
    tag = "_v2" if "--v2" in sys.argv else ""
    for key in [a for a in sys.argv[1:] if not a.startswith("--")]:
        name, src = key.rsplit("_src", 1)
        z = np.load(ROOT / "results/paths" / key / "frames.npz")
        lift = np.load(ROOT / "results/fill" / key / f"lift{tag}.npz")
        kf = lift["kf"] if "kf" in lift.files else None
        g = load_gaussians(ROOT / "results/sharp" / name / f"src{int(src):03d}.pt")
        n = len(lift["xyz"])
        col = torch.from_numpy(srgb_to_linear(lift["rgb"].astype(np.float32) / 255)).float().to(DEV)
        g2 = {
            "means": torch.cat([g["means"], torch.from_numpy(lift["xyz"]).to(DEV)]),
            "quats": torch.cat([g["quats"], torch.tensor([1.0, 0, 0, 0], device=DEV).expand(n, 4)]),
            "scales": torch.cat([g["scales"], torch.from_numpy(lift["radius"]).to(DEV)[:, None].expand(n, 3)]),
            "opacities": torch.cat([g["opacities"], torch.ones(n, device=DEV)]),
            "colors": torch.cat([g["colors"], col]),
            "color_space": "linearRGB",
        }
        K = z["K"].astype(np.float64)
        H, W = z["rgb"].shape[1:3]
        # Background layer: the fill points alone, rendered as sRGB.
        fill_layer = {"means": g2["means"][-n:], "quats": g2["quats"][-n:], "scales": g2["scales"][-n:],
                      "opacities": g2["opacities"][-n:], "colors": torch.from_numpy(lift["rgb"].astype(np.float32) / 255).to(DEV),
                      "color_space": "sRGB"}
        frames, layered, holes = [], [], []
        kernel = np.ones((5, 5), np.uint8)
        with torch.no_grad():
            for i, w2c in enumerate(z["w2c"]):
                rgb, _, alpha = render(g2, w2c.astype(np.float64), K, W, H)
                img = (rgb.cpu().numpy() * 255).astype(np.uint8)
                hole = (alpha < 0.5).cpu().numpy().astype(np.uint8)
                frames.append(cv2.inpaint(img, hole, 5, cv2.INPAINT_TELEA) if hole.any() else img)
                holes.append(hole.mean())
                # Layered composite: SHARP wins wherever the frame's geometry says the photo saw it;
                # the 3D fill layer shows only inside this frame's (slightly dilated) unseen mask.
                layer = fill_layer
                if kf is not None:  # use only the nearest keyframe's fill for this frame
                    near = np.unique(kf)[np.argmin(np.abs(np.unique(kf) - i))]
                    sel = torch.from_numpy(kf == near).to(DEV)
                    layer = {k: (v[sel] if torch.is_tensor(v) else v) for k, v in fill_layer.items()}
                frgb, _, falpha = render(layer, w2c.astype(np.float64), K, W, H)
                unseen = cv2.dilate(z["unseen"][i].astype(np.uint8), kernel) > 0
                use_fill = unseen & (falpha.cpu().numpy() > 0.5)
                out = np.where(use_fill[..., None], (frgb.cpu().numpy() * 255).astype(np.uint8), z["rgb"][i])
                left = (unseen & ~use_fill).astype(np.uint8)
                layered.append(cv2.inpaint(out, left, 5, cv2.INPAINT_TELEA) if left.any() else out)
        np.savez_compressed(ROOT / "results/fill" / key / f"lifted{tag}.npz", rgb=np.stack(frames))
        np.savez_compressed(ROOT / "results/fill" / key / f"layered{tag}.npz", rgb=np.stack(layered))
        print("lifted", key, "points", n, "remaining hole share", np.round(np.mean(holes), 4), flush=True)


if __name__ == "__main__":
    main()
