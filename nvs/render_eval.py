"""Render single-image Gaussian predictions at held-out LLFF views and score them (CUDA / gsplat).

Every method writes Gaussians in the *source camera frame* (OpenCV, identity extrinsics) to
results/<method>/<scene>/srcNNN.pt. Scale is unknown relative to COLMAP, so per source we render
the source-view depth, fit one scalar against COLMAP sparse points, and divide the relative
translation by it.

Covisibility masks are defined once from the SHARP geometry (results/masks) and reused for every
method so masked metrics compare like with like.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gsplat
import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from nvs.scenes import M360, SCENES, load_scene, sparse_depth, target_indices

ROOT = Path(__file__).resolve().parents[1]
DEV = torch.device("cuda")


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)


def load_gaussians(path: Path) -> dict:
    d = torch.load(path, map_location="cpu")
    g = {k: d[k].to(DEV).float() for k in ["means", "quats", "scales", "opacities", "colors"]}
    g["color_space"] = d.get("color_space", "linearRGB")
    return g


@torch.no_grad()
def render(g: dict, w2c: np.ndarray, K: np.ndarray, w: int, h: int):
    colors, alphas, _ = gsplat.rasterization(
        means=g["means"],
        quats=g["quats"],
        scales=g["scales"],
        opacities=g["opacities"],
        colors=g["colors"],
        viewmats=torch.from_numpy(w2c).float().to(DEV)[None],
        Ks=torch.from_numpy(K).float().to(DEV)[None],
        width=w,
        height=h,
        render_mode="RGB+D",
        rasterize_mode="classic",
    )
    rgb, depth, alpha = colors[0, ..., :3], colors[0, ..., 3], alphas[0, ..., 0]
    depth = depth / alpha.clamp(min=1e-6)
    if g["color_space"] == "linearRGB":
        rgb = linear_to_srgb(rgb)
    return rgb.clamp(0, 1), depth, alpha


def covis_mask(depth_t, alpha_t, depth_s, T_t_from_s, K_t, K_s, w, h, tol=0.05):
    """Target pixels whose predicted surface point is inside the source frame and unoccluded there."""
    ys, xs = torch.meshgrid(torch.arange(h, device=DEV), torch.arange(w, device=DEV), indexing="ij")
    pix = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs, dtype=torch.float32)], -1).float()
    Kt_inv = torch.from_numpy(np.linalg.inv(K_t)).float().to(DEV)
    X_t = (pix @ Kt_inv.T) * depth_t[..., None]
    T = torch.from_numpy(np.linalg.inv(T_t_from_s)).float().to(DEV)
    X_s = X_t @ T[:3, :3].T + T[:3, 3]
    z = X_s[..., 2]
    uv = X_s @ torch.from_numpy(K_s).float().to(DEV).T
    u, v = uv[..., 0] / z.clamp(min=1e-6), uv[..., 1] / z.clamp(min=1e-6)
    inside = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ui, vi = u.long().clamp(0, w - 1), v.long().clamp(0, h - 1)
    unoccluded = z <= depth_s[vi, ui] * (1 + tol)
    return inside & unoccluded & (alpha_t > 0.5)


def inpaint_holes(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    import cv2

    img = (rgb.cpu().numpy() * 255).astype(np.uint8)
    holes = (alpha < 0.5).cpu().numpy().astype(np.uint8)
    out = cv2.inpaint(img, holes, 5, cv2.INPAINT_TELEA)
    return torch.from_numpy(out).float().to(rgb.device) / 255


def psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    se = ((a - b) ** 2).mean(-1)
    mse = se.mean() if mask is None else (se[mask].mean() if mask.any() else torch.tensor(float("nan")))
    return float(-10 * torch.log10(mse.clamp(min=1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="sharp")
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--fill", action="store_true", help="inpaint alpha<0.5 holes (Telea) before scoring")
    ap.add_argument("--own-masks", action="store_true", help="covisibility from this method's own geometry (not SHARP's)")
    args = ap.parse_args()
    if args.scenes == ["m360"]:
        args.scenes = M360
    elif args.scenes == ["all"]:
        args.scenes = SCENES + M360

    lp = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(DEV)
    out_dir = ROOT / "results" / "eval" / (args.method + ("_fill" if args.fill else "") + ("_own" if args.own_masks else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.scenes:
        scene = load_scene(name)
        w, h = scene.width, scene.height
        rows = []
        for src_file in sorted((ROOT / "results" / args.method / name).glob("src*.pt")):
            g = load_gaussians(src_file)
            si = int(src_file.stem[3:])
            sv = scene.views[si]

            _, depth_s, alpha_s = render(g, np.eye(4), sv.K, w, h)
            uv, z = sparse_depth(scene, sv)
            ui = torch.from_numpy(uv[:, 0].astype(np.int64)).to(DEV)
            vi = torch.from_numpy(uv[:, 1].astype(np.int64)).to(DEV)
            dpred = depth_s[vi, ui].cpu().numpy()
            ok = (alpha_s[vi, ui].cpu().numpy() > 0.5) & (dpred > 1e-4)
            ratio = z[ok] / dpred[ok]
            scale = float(np.median(ratio))
            align_err = float(np.median(np.abs(np.log(ratio / scale))))
            zmed = float(np.median(z))

            src_img = torch.from_numpy(np.asarray(Image.open(sv.path).convert("RGB"))).float().to(DEV) / 255
            mask_dir = ROOT / "results" / "masks" / name / f"src{si:03d}"
            mask_dir.mkdir(parents=True, exist_ok=True)

            targets = target_indices(scene, si)
            showcase = _showcase_targets(scene, si, targets)
            for ti in targets:
                tv = scene.views[ti]
                T = tv.w2c @ np.linalg.inv(sv.w2c)  # source cam (COLMAP units) -> target cam
                T_pred = T.copy()
                T_pred[:3, 3] /= scale
                rgb, depth_t, alpha_t = render(g, T_pred, tv.K, w, h)
                if args.fill:
                    rgb = inpaint_holes(rgb, alpha_t)
                gt = torch.from_numpy(np.asarray(Image.open(tv.path).convert("RGB"))).float().to(DEV) / 255

                mask_path = mask_dir / f"tgt{ti:03d}.npy"
                if args.own_masks:
                    mask = covis_mask(depth_t, alpha_t, depth_s, T_pred, tv.K, sv.K, w, h)
                elif args.method == "sharp" and not args.fill:
                    mask = covis_mask(depth_t, alpha_t, depth_s, T_pred, tv.K, sv.K, w, h)
                    np.save(mask_path, np.packbits(mask.cpu().numpy()))
                else:
                    bits = np.unpackbits(np.load(mask_path))[: w * h].reshape(h, w).astype(bool)
                    mask = torch.from_numpy(bits).to(DEV)

                lmap = lp(rgb.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1)[0, 0]
                lmap_id = lp(src_img.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1)[0, 0]
                rgb_np, gt_np = rgb.cpu().numpy(), gt.cpu().numpy()
                cos = (np.trace(T[:3, :3]) - 1) / 2
                rows.append(
                    {
                        "scene": name,
                        "src": si,
                        "tgt": ti,
                        "baseline_rel": float(np.linalg.norm(T[:3, 3]) / zmed),
                        "rot_deg": math.degrees(math.acos(max(-1.0, min(1.0, cos)))),
                        "scale": scale,
                        "align_err": align_err,
                        "covis_frac": float(mask.float().mean()),
                        "alpha_mean": float(alpha_t.mean()),
                        "psnr": psnr(rgb, gt),
                        "psnr_covis": psnr(rgb, gt, mask),
                        "psnr_disocc": psnr(rgb, gt, ~mask),
                        "ssim": float(structural_similarity(rgb_np, gt_np, channel_axis=2, data_range=1.0)),
                        "lpips": float(lmap.mean()),
                        "lpips_covis": float(lmap[mask].mean()) if mask.any() else float("nan"),
                        "lpips_disocc": float(lmap[~mask].mean()) if (~mask).any() else float("nan"),
                        "id_psnr": psnr(src_img, gt),
                        "id_lpips": float(lmap_id.mean()),
                    }
                )
                if ti in showcase:
                    fig_dir = ROOT / "results" / "figs" / out_dir.name / name
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    strip = np.concatenate([gt_np, rgb_np, np.repeat(mask.cpu().numpy()[..., None], 3, -1)], 1)
                    Image.fromarray((strip * 255).astype(np.uint8)).save(fig_dir / f"src{si:03d}_tgt{ti:03d}.jpg", quality=88)
            r = [x for x in rows if x["src"] == si]
            print(
                f"{args.method} {name} src{si:03d} scale={scale:.3f} align_err={align_err:.3f} "
                f"psnr={np.mean([x['psnr'] for x in r]):.2f} lpips={np.mean([x['lpips'] for x in r]):.3f} "
                f"id_psnr={np.mean([x['id_psnr'] for x in r]):.2f}",
                flush=True,
            )
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def _showcase_targets(scene, si: int, targets: list[int]) -> set[int]:
    c = np.stack([v.center for v in scene.views])
    order = sorted(targets, key=lambda i: np.linalg.norm(c[i] - c[si]))
    return {order[0], order[len(order) // 2], order[-1]}


if __name__ == "__main__":
    main()
