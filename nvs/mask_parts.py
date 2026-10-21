"""Split SHARP's unseen mask into out-of-frame vs occluded parts, per eval pair (desktop / CUDA).

No image metrics: renders depth/alpha only and writes results/mask_parts/<scene>.jsonl with
out_of_frame_frac (target pixels that project outside the source frame), occluded_frac (inside the
frame but behind a nearer source surface), and hole_frac (alpha < 0.5).
"""

from __future__ import annotations

import json

import numpy as np
import torch

from nvs.render_eval import DEV, ROOT, load_gaussians, render
from nvs.scenes import M360, SCENES, load_scene, sparse_depth, target_indices


def parts(depth_t, alpha_t, depth_s, T_t_from_s, K_t, K_s, w, h, tol=0.05):
    ys, xs = torch.meshgrid(torch.arange(h, device=DEV), torch.arange(w, device=DEV), indexing="ij")
    pix = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs, dtype=torch.float32)], -1).float()
    X_t = (pix @ torch.from_numpy(np.linalg.inv(K_t)).float().to(DEV).T) * depth_t[..., None]
    T = torch.from_numpy(np.linalg.inv(T_t_from_s)).float().to(DEV)
    X_s = X_t @ T[:3, :3].T + T[:3, 3]
    z = X_s[..., 2]
    uv = X_s @ torch.from_numpy(K_s).float().to(DEV).T
    u, v = uv[..., 0] / z.clamp(min=1e-6), uv[..., 1] / z.clamp(min=1e-6)
    solid = alpha_t > 0.5
    inside = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ui, vi = u.long().clamp(0, w - 1), v.long().clamp(0, h - 1)
    occluded = inside & (z > depth_s[vi, ui] * (1 + tol))
    return {
        "out_of_frame_frac": float((solid & ~inside).float().mean()),
        "occluded_frac": float((solid & occluded).float().mean()),
        "hole_frac": float((~solid).float().mean()),
    }


def main():
    import sys

    tol = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    out_dir = ROOT / "results" / ("mask_parts" if tol == 0.05 else f"mask_parts_tol{tol:g}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SCENES + M360:
        scene = load_scene(name)
        w, h = scene.width, scene.height
        rows = []
        for src_file in sorted((ROOT / "results" / "sharp" / name).glob("src*.pt")):
            g = load_gaussians(src_file)
            si = int(src_file.stem[3:])
            sv = scene.views[si]
            with torch.no_grad():
                _, depth_s, alpha_s = render(g, np.eye(4), sv.K, w, h)
                uv, z = sparse_depth(scene, sv)
                ui = torch.from_numpy(uv[:, 0].astype(np.int64)).to(DEV)
                vi = torch.from_numpy(uv[:, 1].astype(np.int64)).to(DEV)
                d = depth_s[vi, ui].cpu().numpy()
                ok = (alpha_s[vi, ui].cpu().numpy() > 0.5) & (d > 1e-4)
                scale = float(np.median(z[ok] / d[ok]))
                for ti in target_indices(scene, si):
                    tv = scene.views[ti]
                    T = tv.w2c @ np.linalg.inv(sv.w2c)
                    T[:3, 3] /= scale
                    _, depth_t, alpha_t = render(g, T, tv.K, w, h)
                    rows.append({"scene": name, "src": si, "tgt": ti, **parts(depth_t, alpha_t, depth_s, T, tv.K, sv.K, w, h, tol)})
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("parts", name, len(rows), flush=True)


if __name__ == "__main__":
    main()
