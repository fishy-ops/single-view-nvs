"""Compare nvs/splat_torch.render_torch against gsplat on a real scene (run on the desktop)."""
from __future__ import annotations
import sys, time
import numpy as np, torch
from nvs.render_eval import DEV, ROOT, load_gaussians, render
from nvs.scenes import load_scene
from nvs.splat_torch import render_torch

name, si = sys.argv[1].split(":")
scene = load_scene(name); sv = scene.views[int(si)]
g = load_gaussians(ROOT / "results/sharp" / name / f"src{int(si):03d}.pt")
W, H = 256, int(round(scene.height * 256 / scene.width))
K = sv.K.copy(); K[:2] *= W / scene.width
Kt = torch.from_numpy(K).float().to(DEV)
w2c = torch.eye(4, device=DEV)
t0 = time.time(); ref, refd, refa = render(g, np.eye(4), K, W, H); t_gsplat = time.time() - t0
t0 = time.time()
with torch.no_grad():
    out, d, a = render_torch(g["means"], g["quats"], g["scales"], g["opacities"], g["colors"], w2c, Kt, W, H)
t_torch = time.time() - t0
from nvs.render_eval import linear_to_srgb
out = linear_to_srgb(out).clamp(0, 1)
err = (out - ref).abs()
print(f"{name}:{si} {W}x{H} gsplat {t_gsplat:.2f}s torch {t_torch:.2f}s")
print(f"  colour  mean|diff| {err.mean():.4f}  p99 {err.flatten().kthvalue(int(0.99*err.numel())).values:.4f}")
print(f"  alpha   mean|diff| {(a-refa).abs().mean():.4f}   depth mean rel {(((d-refd).abs())/(refd.clamp(min=1e-3))).median():.4f}")
