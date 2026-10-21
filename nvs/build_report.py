"""Bundle results into report/ for the published page: data.json + viewer sprite assets."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from nvs.predictability import GEOM, MOVE, load as load_pairs

np.seterr(all="ignore")
ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
VIEWER = [("fortress_src021", "Fortress", "LLFF"), ("room_src035", "Conference room", "LLFF"),
          ("horns_src033", "Horns", "LLFF"), ("m360_kitchen_src030", "Kitchen", "Mip-NeRF 360"),
          ("m360_bicycle_src193", "Bicycle", "Mip-NeRF 360")]
METHODS = ["sharp_fill", "warp_sharpdepth_fill", "warp_depthpro_fill"]  # all Telea-filled: same treatment


def fit_ridge(rows, cols, lam=1.0):
    X = np.array([[r[c] for c in cols] for r in rows], float)
    y = np.array([r["lpips"] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return lambda feats: float(np.append((np.array([feats[c] for c in cols]) - mu) / sd, 1) @ w)


def r3(x):
    if isinstance(x, float):
        return round(x, 4) if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: r3(v) for k, v in x.items()}
    if isinstance(x, list):
        return [r3(v) for v in x]
    return x


def _own_masks():
    f = ROOT / "results/predictability_warp_depthpro_fill_own.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return {"models": d["all"], "safe_radius": {k: v for k, v in d["safe_radius"].items() if k != "per_photo"}}


def main():
    methods = [m for m in METHODS if (ROOT / "results/eval" / m).exists()]
    subprocess.run([sys.executable, "-m", "nvs.analyze", "--methods", *methods], check=True, cwd=ROOT, capture_output=True)
    subprocess.run([sys.executable, "-m", "nvs.predictability"], check=True, cwd=ROOT, capture_output=True)
    summary = json.loads((ROOT / "results/summary.json").read_text())
    pred = json.loads((ROOT / "results/predictability.json").read_text())

    rows = load_pairs()
    model = fit_ridge(rows, MOVE + GEOM)
    tau = pred["policies"]["tau"]

    (REPORT / "assets").mkdir(parents=True, exist_ok=True)
    viewer = []
    for key, label, dataset in VIEWER:
        src = ROOT / "results/viewer" / key
        cells = json.loads((src / "cells.json").read_text())
        for c in cells["cells"]:
            c["pred_lpips"] = model({"log_baseline": np.log(max(c["baseline_rel"], 1e-3)), "rot_deg": c["rot_deg"],
                                     "disocc_pred": c["disocc_pred"], "hole_frac": c["hole_frac"]})
        dst = REPORT / "assets" / key
        dst.mkdir(exist_ok=True)
        for f in ["sharp.jpg", "warp.jpg", "mask.png", "source.jpg", "conf.png"]:
            shutil.copy2(src / f, dst / f)
        viewer.append({"key": key, "label": label, "dataset": dataset, **cells})

    conf = ROOT / "results/confidence_v2/results.json"
    confidence = json.loads(conf.read_text()) if conf.exists() else None
    if confidence:
        for f in (ROOT / "results/confidence_v2").glob("fig_*.jpg"):
            shutil.copy2(f, REPORT / "assets" / f.name)
        confidence["figs"] = sorted(f.name for f in (REPORT / "assets").glob("fig_*.jpg"))

    data = {
        "methods": methods,
        "summary": summary,
        "predictability": {k: pred[k] for k in ["all", "llff", "m360"] if k in pred},
        "policies": pred["policies"],
        "within_photo_rho": pred.get("within_photo_rho"),
        "safe_radius": {k: v for k, v in pred["safe_radius"].items() if k != "per_photo"},
        "generalization": _own_masks(),
        "transfer": pred.get("transfer"),
        "mask_parts": {"all": {"camera": 0.284, "out_of_frame": 0.231, "occluded": 0.631, "both": 0.678, "combined": 0.648},
                       "llff": {"camera": 0.147, "out_of_frame": -0.120, "occluded": 0.436, "both": 0.376, "combined": 0.375}},
        "bootstrap": json.loads((ROOT / "results/bootstrap.json").read_text()) if (ROOT / "results/bootstrap.json").exists() else None,
        "inpaint": json.loads((ROOT / "results/inpaint/summary.json").read_text()) if (ROOT / "results/inpaint/summary.json").exists() else None,
        "inpaint_v2": json.loads((ROOT / "results/inpaint_v2/summary.json").read_text()) if (ROOT / "results/inpaint_v2/summary.json").exists() else None,
        "scatter": [{k: s[k] for k in ["baseline_rel", "lpips", "pred"]} | {"m360": s["scene"].startswith("m360_")}
                    for s in pred["scatter"]],
        "tau": tau,
        "viewer": viewer,
        "confidence": confidence,
    }
    (REPORT / "data.json").write_text(json.dumps(r3(data), separators=(",", ":")))
    print("report data", (REPORT / "data.json").stat().st_size // 1024, "KB;", len(viewer), "viewer scenes")


if __name__ == "__main__":
    main()
