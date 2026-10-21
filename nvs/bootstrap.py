"""Scene-level bootstrap CIs for headline numbers (Mac, numpy only).

Held-out (LOSO) predictions are computed once; each bootstrap resamples the 15 scenes with
replacement and recomputes the evaluation metrics on the resampled rows. Pairs within a scene are
not independent, so resampling scenes (not pairs) gives honest intervals.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nvs.predictability import GEOM, MOVE, load, loso, spearman

np.seterr(all="ignore")
ROOT = Path(__file__).resolve().parents[1]
B = 1000


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def within_bin(y, p, b):
    edges = np.quantile(b, np.linspace(0, 1, 6))
    bins = np.clip(np.searchsorted(edges, b, side="right") - 1, 0, 4)
    return float(np.nanmean([spearman(y[bins == i], p[bins == i]) for i in range(5) if (bins == i).sum() > 5]))


def radius_spearman(rows, y, p, tau, idx):
    keys = np.array([f"{rows[i]['scene']}/{rows[i]['src']}" for i in idx])
    lb = np.log([rows[i]["baseline_rel"] for i in idx])
    yy, pp = y[idx], p[idx]
    rm, rp = [], []
    for k in np.unique(keys):
        m = keys == k
        if m.sum() < 8:
            continue
        A = np.stack([np.ones(m.sum()), lb[m]], 1)
        (am, bm), (ap, bp) = np.linalg.lstsq(A, yy[m], rcond=None)[0], np.linalg.lstsq(A, pp[m], rcond=None)[0]
        rad = lambda a, b: float(np.clip(np.exp((tau - a) / b), 0.005, 2.0)) if b > 1e-4 else 2.0
        rm.append(rad(am, bm))
        rp.append(rad(ap, bp))
    return spearman(np.array(rp), np.array(rm))


def ci(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    rows = load("sharp")
    y, p_geo = loso(rows, MOVE + GEOM)
    _, p_cam = loso(rows, MOVE)
    b = np.array([r["baseline_rel"] for r in rows])
    scenes = np.array([r["scene"] for r in rows])
    uniq = np.unique(scenes)
    tau = float(np.median(y))
    by_scene = {s: np.where(scenes == s)[0] for s in uniq}

    # SHARP vs pixel warp (both filled): paired LPIPS on the 1350 jointly scored pairs.
    def eval_rows(m):
        out = {}
        for f in sorted((ROOT / "results/eval" / m).glob("*.jsonl")):
            for r in map(json.loads, f.read_text().splitlines()):
                out[(r["scene"], r["src"], r["tgt"])] = r["lpips"]
        return out
    sf, wf = eval_rows("sharp_fill"), eval_rows("warp_sharpdepth_fill")
    keys = sorted(set(sf) & set(wf))
    d_scene = {s: np.array([wf[k] - sf[k] for k in keys if k[0] == s]) for s in uniq}

    rng = np.random.default_rng(0)
    stats = {k: [] for k in ["r2_cam", "r2_geo", "wb_cam", "wb_geo", "radius_rho", "bad50_fixed", "bad50_pred", "d_lpips_warp", "win_warp"]}
    point = {}
    for it in range(B + 1):
        pick = uniq if it == 0 else rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_scene[s] for s in pick])
        yy, pg, pc, bb = y[idx], p_geo[idx], p_cam[idx], b[idx]
        k = len(idx) // 2
        vals = {
            "r2_cam": r2(yy, pc), "r2_geo": r2(yy, pg),
            "wb_cam": within_bin(yy, pc, bb), "wb_geo": within_bin(yy, pg, bb),
            "radius_rho": radius_spearman(rows, y, p_geo, tau, idx),
            "bad50_fixed": float((yy[np.argsort(bb)[:k]] > tau).mean()),
            "bad50_pred": float((yy[np.argsort(pg)[:k]] > tau).mean()),
            "d_lpips_warp": float(np.concatenate([d_scene[s] for s in pick]).mean()),
            "win_warp": float((np.concatenate([d_scene[s] for s in pick]) > 0).mean()),
        }
        if it == 0:
            point = vals
        else:
            for kk, v in vals.items():
                stats[kk].append(v)
    for name, (x_, y_) in {"r2_gain": ("r2_geo", "r2_cam"), "wb_gain": ("wb_geo", "wb_cam"), "bad50_reduction": ("bad50_fixed", "bad50_pred")}.items():
        stats[name] = list(np.array(stats[x_]) - np.array(stats[y_]))
        point[name] = point[x_] - point[y_]
    out = {kk: {"point": point[kk], "ci95": ci(stats[kk])} for kk in stats}
    out["n_bootstrap"] = B
    out["unit"] = "scene"
    (ROOT / "results/bootstrap.json").write_text(json.dumps(out, indent=1))
    for kk in stats:
        print(f"{kk:14s} {point[kk]:.3f}  [{out[kk]['ci95'][0]:.3f}, {out[kk]['ci95'][1]:.3f}]")


if __name__ == "__main__":
    main()
