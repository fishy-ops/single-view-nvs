"""Diagnostic: is the relative pose convention right? Sweep sign/transpose/scale variants on one source."""
import sys
import numpy as np, torch
from PIL import Image
from nvs.render_eval import render, load_gaussians, psnr, DEV, ROOT
from nvs.scenes import load_scene

name, si = sys.argv[1], int(sys.argv[2])
sc = load_scene(name); sv = sc.views[si]
g = load_gaussians(ROOT / "results" / "sharp" / name / f"src{si:03d}.pt")
img = lambda v: torch.from_numpy(np.asarray(Image.open(v.path).convert("RGB"))).float().to(DEV) / 255
rgb, _, _ = render(g, np.eye(4), sv.K, sc.width, sc.height)
print("self-render psnr", psnr(rgb, img(sv)))
c = np.stack([v.center for v in sc.views]); order = [int(i) for i in np.argsort(np.linalg.norm(c - c[si], axis=1)) if i != si][:3]
scale = float(sys.argv[3])
for ti in order:
    tv = sc.views[ti]; gt = img(tv)
    T = tv.w2c @ np.linalg.inv(sv.w2c)
    res = {}
    for label, TT in {"T": T, "inv(T)": np.linalg.inv(T), "R only": np.block([[T[:3,:3], np.zeros((3,1))],[np.zeros((1,3)),1]]), "t only": np.block([[np.eye(3), T[:3,3:]],[np.zeros((1,3)),1]])}.items():
        for sm in [0.5, 1.0, 2.0]:
            X = TT.copy(); X[:3, 3] /= (scale * sm)
            res[f"{label} s*{sm}"] = psnr(render(g, X, tv.K, sc.width, sc.height)[0], gt)
    best = sorted(res.items(), key=lambda kv: -kv[1])[:4]
    print(ti, "identity", round(psnr(img(sv), gt), 2), "|", ", ".join(f"{k}={v:.2f}" for k, v in best), "| T s*1", round(res["T s*1.0"], 2))
