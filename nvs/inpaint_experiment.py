"""Are confidence maps actionable? Inpaint SHARP renders where the map says, vs. baselines (Mac).

Per test pair (up to 40 per scene, spread over camera distance), with the same pixel budget
a = share of pixels SHARP's geometry marks unseen (clipped to [2%, 50%]):
  unseen : inpaint the unseen mask
  conf   : inpaint the top-a pixels of the held-out confidence UNet's predicted error
  random : inpaint the top-a pixels of a *different* pair's predicted error (same blob statistics)
and "raw" = no inpainting. LaMa (big-lama) does the inpainting. Scored with LPIPS (alex) and PSNR
against the real photo. Output: results/inpaint/rows.jsonl and summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama

from nvs.train_confidence import ROOT, UNet, folds

import os

OUT = ROOT / "results" / os.environ.get("INPAINT_OUT", "inpaint")
VARIANTS = os.environ.get("INPAINT_VARIANTS", "unseen,conf,random").split(",")
RUN = "confidence_v2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "results/pairs").glob("*/*.npz"))
    by_scene = {}
    for f in files:
        by_scene.setdefault(f.parent.name, []).append(f)
    fold_of = folds([{"scene": s} for s in by_scene])
    dev = torch.device("mps")
    models = {}
    for k in sorted(set(fold_of.values())):
        m = UNet().to(dev).eval()
        m.load_state_dict(torch.load(ROOT / "results" / RUN / f"unet_fold{k}.pt", map_location=dev))
        models[k] = m
    try:
        lama = SimpleLama(device=torch.device("mps"))
        lama_dev = "mps"
    except Exception as e:  # torchscript ops missing on MPS
        print("LaMa on MPS failed, using CPU:", type(e).__name__)
        lama, lama_dev = SimpleLama(device=torch.device("cpu")), "cpu"
    lp = lpips.LPIPS(net="alex", verbose=False)
    rng = np.random.default_rng(0)

    chosen = []
    for s, fs in sorted(by_scene.items()):
        metas = [(json.loads(str(np.load(f)["meta"]))["baseline_rel"], f) for f in fs]
        metas.sort(key=lambda t: t[0])
        idx = np.linspace(0, len(metas) - 1, min(40, len(metas))).round().astype(int)
        chosen += [metas[i][1] for i in sorted(set(idx))]
    print(f"{len(chosen)} pairs, LaMa on {lama_dev}", flush=True)

    # Precompute predicted error maps (needed for the random-control shuffle).
    preds = {}
    for f in chosen:
        z = np.load(f)
        x = np.concatenate([z["render"].astype(np.float32) / 255, (z["alpha"].astype(np.float32) / 255)[..., None],
                            (np.clip(z["depth_rel"].astype(np.float32), -3, 3) / 3)[..., None],
                            (z["unseen"].astype(np.float32) / 255)[..., None]], -1)
        with torch.no_grad():
            preds[f] = models[fold_of[f.parent.name]](torch.from_numpy(x).permute(2, 0, 1)[None].to(dev))[0, 0].cpu().numpy()

    def score(img, gt):
        a = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 127.5 - 1
        b = torch.from_numpy(gt).permute(2, 0, 1)[None].float() / 127.5 - 1
        mse = float(((img.astype(np.float32) - gt.astype(np.float32)) ** 2).mean()) / 255 ** 2
        return float(lp(a, b)), float(-10 * np.log10(max(mse, 1e-10)))

    def inpaint(render, mask):
        out = np.asarray(lama(Image.fromarray(render), Image.fromarray((mask * 255).astype(np.uint8))))
        return out[: render.shape[0], : render.shape[1]]

    done = set()
    rows_path = OUT / "rows.jsonl"
    if rows_path.exists():
        done = {r["file"] for r in map(json.loads, rows_path.read_text().splitlines())}
    with open(rows_path, "a") as fh:
        for n, f in enumerate(chosen):
            if str(f) in done:
                continue
            z = np.load(f)
            render, gt = z["render"], z["gt"]
            unseen = z["unseen"] > 127
            a = float(np.clip(unseen.mean(), 0.02, 0.5))
            k = max(1, int(a * unseen.size))

            def top(m):
                thr = np.partition(m.reshape(-1), -k)[-k]
                return m >= thr

            other = chosen[rng.integers(len(chosen))]
            om = preds[other]
            if om.shape != render.shape[:2]:
                om = np.asarray(Image.fromarray(om).resize((render.shape[1], render.shape[0])))
            from scipy.ndimage import binary_dilation

            k5 = max(1, int(0.05 * unseen.size))
            thr5 = np.partition(preds[f].reshape(-1), -k5)[-k5]
            all_masks = {"unseen": unseen if unseen.mean() >= 0.02 else top(unseen.astype(np.float32) + 1e-3 * preds[f]),
                         "conf": top(preds[f]), "random": top(om),
                         "unseen_dil": binary_dilation(unseen, iterations=6),
                         "unseen_plus_conf": unseen | (preds[f] >= thr5)}
            masks = {v: all_masks[v] for v in VARIANTS}
            row = {"file": str(f), "scene": f.parent.name, **json.loads(str(z["meta"])), "budget": a}
            row["raw_lpips"], row["raw_psnr"] = score(render, gt)
            for name, m in masks.items():
                row[f"{name}_lpips"], row[f"{name}_psnr"] = score(inpaint(render, m.astype(np.uint8)), gt)
                row[f"{name}_area"] = float(m.mean())
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n % 50 == 0:
                print(n, row["scene"], {k2: round(v, 3) for k2, v in row.items() if k2.endswith("_lpips")}, flush=True)

    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    summ = {"n": len(rows)}
    for name in VARIANTS:
        d = np.array([r["raw_lpips"] - r[f"{name}_lpips"] for r in rows])
        dp = np.array([r[f"{name}_psnr"] - r["raw_psnr"] for r in rows])
        summ[name] = {"d_lpips_mean": float(d.mean()), "lpips_better_rate": float((d > 0).mean()), "d_psnr_mean": float(dp.mean()),
                      "mean_area": float(np.mean([r[f"{name}_area"] for r in rows]))}
    for a_, b_ in [("conf", "unseen"), ("conf", "random"), ("unseen_plus_conf", "unseen"), ("unseen_dil", "unseen"), ("unseen_plus_conf", "unseen_dil")]:
        if a_ in VARIANTS and b_ in VARIANTS:
            summ[f"{a_}_vs_{b_}_lpips_win"] = float(np.mean([r[f"{a_}_lpips"] < r[f"{b_}_lpips"] for r in rows]))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print("SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
