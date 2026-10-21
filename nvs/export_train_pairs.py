"""Training pairs for the background-layer head (desktop / CUDA).

Per (scene, source, target): the source photo and SHARP's source-view depth/alpha (the head's
input), the relative camera and intrinsics, and the target view's real photo, SHARP render, alpha,
depth and unseen mask (the supervision). Everything at width 384, in the source camera frame, so
the Mac can train without CUDA. Output: results/train_pairs/<scene>/s<src>_t<tgt>.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from PIL import Image

from nvs.render_eval import DEV, ROOT, covis_mask, load_gaussians, render
from nvs.scenes import M360, SCENES, load_scene, sparse_depth, target_indices

OUT_W = 384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=SCENES + M360)
    args = ap.parse_args()
    for name in args.scenes:
        scene = load_scene(name)
        s = OUT_W / scene.width
        W, H = OUT_W, int(round(scene.height * s)) // 8 * 8
        out_dir = ROOT / "results" / "train_pairs" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for src_file in sorted((ROOT / "results" / "sharp" / name).glob("src*.pt")):
            si = int(src_file.stem[3:])
            sv = scene.views[si]
            K = sv.K.copy()
            K[:2] *= s
            g = load_gaussians(src_file)
            with torch.no_grad():
                src_rgb, depth_s, alpha_s = render(g, np.eye(4), K, W, H)
                uv, z = sparse_depth(scene, sv)
                _, dfull, afull = render(g, np.eye(4), sv.K, scene.width, scene.height)
                ui = torch.from_numpy(uv[:, 0].astype(np.int64)).to(DEV)
                vi = torch.from_numpy(uv[:, 1].astype(np.int64)).to(DEV)
                d = dfull[vi, ui].cpu().numpy()
                ok = (afull[vi, ui].cpu().numpy() > 0.5) & (d > 1e-4)
                scale = float(np.median(z[ok] / d[ok]))
                for ti in target_indices(scene, si):
                    path = out_dir / f"s{si:03d}_t{ti:03d}.npz"
                    if path.exists():
                        continue
                    tv = scene.views[ti]
                    Kt = tv.K.copy()
                    Kt[:2] *= s
                    T = tv.w2c @ np.linalg.inv(sv.w2c)
                    T[:3, 3] /= scale  # into SHARP's units
                    rgb, depth_t, alpha_t = render(g, T, Kt, W, H)
                    unseen = ~covis_mask(depth_t, alpha_t, depth_s, T, Kt, K, W, H)
                    gt = torch.from_numpy(np.asarray(Image.open(tv.path).convert("RGB").resize((W, H)))).to(DEV)
                    np.savez_compressed(
                        path,
                        src_rgb=(src_rgb.clamp(0, 1) * 255).byte().cpu().numpy(),
                        src_depth=depth_s.half().cpu().numpy(),
                        src_alpha=(alpha_s.clamp(0, 1) * 255).byte().cpu().numpy(),
                        tgt_gt=gt.byte().cpu().numpy(),
                        tgt_render=(rgb.clamp(0, 1) * 255).byte().cpu().numpy(),
                        tgt_alpha=(alpha_t.clamp(0, 1) * 255).byte().cpu().numpy(),
                        tgt_depth=depth_t.half().cpu().numpy(),
                        unseen=unseen.cpu().numpy(),
                        K=K.astype(np.float32), K_tgt=Kt.astype(np.float32), w2c=T.astype(np.float32),
                        meta=json.dumps({"scene": name, "src": si, "tgt": ti, "scale": scale}),
                    )
                    n += 1
        print("train_pairs", name, n, flush=True)


if __name__ == "__main__":
    main()
