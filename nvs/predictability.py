"""Can we predict novel-view quality from the single input image + requested camera move?

Per (scene, src, tgt) target: SHARP LPIPS at the target. Features are available at test time:
the camera move, SHARP-predicted disocclusion (1 - covis_frac, from SHARP geometry at the target
pose) and hole fraction, and per-source image/geometry stats (results/source_features.jsonl).

Ridge regression, leave-one-scene-out. Reports R^2, pooled Spearman, and Spearman *within*
baseline bins (does it rank images correctly at the same camera move? A fixed movement limit
cannot do this.)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

np.seterr(all="ignore")  # macOS Accelerate raises spurious matmul FPE warnings

ROOT = Path(__file__).resolve().parents[1]
MOVE = ["log_baseline", "rot_deg"]
GEOM = ["disocc_pred", "hole_frac"]
SRC = ["self_lpips", "depth_edge_frac", "depth_log_range", "texture", "opacity_lowfrac"]


def rank(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1]) if len(a) > 2 else float("nan")


def load(method="sharp"):
    feats = {(r["scene"], r["src"]): r for r in map(json.loads, (ROOT / "results/source_features.jsonl").read_text().splitlines())}
    rows = []
    for f in sorted((ROOT / "results/eval" / method).glob("*.jsonl")):
        for r in map(json.loads, f.read_text().splitlines()):
            sf = feats.get((r["scene"], r["src"]))
            if sf is None or not np.isfinite(r["lpips"]):
                continue
            rows.append({**sf, **r, "log_baseline": np.log(r["baseline_rel"]), "disocc_pred": 1 - r["covis_frac"],
                         "hole_frac": 1 - r["alpha_mean"]})
    return rows


def design(rows, cols, interact):
    X = np.array([[r[c] for c in cols] for r in rows], float)
    if interact:  # image-dependent slope: source features x camera move
        lb = np.array([r["log_baseline"] for r in rows])[:, None]
        X = np.hstack([X, X[:, [cols.index(c) for c in cols if c in SRC + GEOM]] * lb])
    return X


def loso(rows, cols, interact=False, lam=1.0):
    y = np.array([r["lpips"] for r in rows])
    X = design(rows, cols, interact)
    scenes = np.array([r["scene"] for r in rows])
    pred = np.zeros_like(y)
    for s in np.unique(scenes):
        tr, te = scenes != s, scenes == s
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.hstack([(X[tr] - mu) / sd, np.ones((tr.sum(), 1))])
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.hstack([(X[te] - mu) / sd, np.ones((te.sum(), 1))]) @ w
    return y, pred


def report(rows):
    b = np.array([r["baseline_rel"] for r in rows])
    edges = np.quantile(b, np.linspace(0, 1, 6))
    bins = np.clip(np.searchsorted(edges, b, side="right") - 1, 0, 4)
    out = {}
    models = {"camera move only": (MOVE, False), "+ predicted disocclusion": (MOVE + GEOM, False),
              "+ source image stats": (MOVE + GEOM + SRC, False), "+ image-dependent slope": (MOVE + GEOM + SRC, True)}
    print(f"{len(rows)} pairs, {len({r['scene'] for r in rows})} scenes, leave-one-scene-out ridge")
    print(f"{'model':28s} {'R2':>6s} {'rho':>6s} {'rho|bin':>8s}")
    for name, (cols, inter) in models.items():
        y, p = loso(rows, cols, inter)
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        within = np.nanmean([spearman(y[bins == i], p[bins == i]) for i in range(5)])
        out[name] = {"r2": float(r2), "spearman": spearman(y, p), "spearman_within_bin": float(within)}
        print(f"{name:28s} {r2:6.3f} {spearman(y, p):6.3f} {within:8.3f}")
    return out


def policies(rows, tau=None):
    """Fixed movement limit vs predicted-quality limit, at equal coverage (share of views allowed).

    Fixed: allow the q-fraction of pairs with the smallest camera move. Adaptive: allow the
    q-fraction with the lowest LOSO-predicted LPIPS (camera move + predicted disocclusion).
    Bad view = actual LPIPS above tau (default: overall median).
    """
    y, p = loso(rows, MOVE + GEOM)
    b = np.array([r["baseline_rel"] for r in rows])
    tau = float(np.median(y)) if tau is None else tau
    out = {"tau": tau, "coverage": [], "fixed_bad": [], "adaptive_bad": [], "oracle_bad": [],
           "fixed_mean": [], "adaptive_mean": []}
    print(f"\npolicy comparison: bad = LPIPS > {tau:.3f}")
    print(f"{'allowed':>8s} {'fixed bad%':>10s} {'adaptive bad%':>13s} {'oracle bad%':>11s} {'fixed LPIPS':>11s} {'adapt LPIPS':>11s}")
    for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        k = int(round(q * len(y)))
        f, a, o = np.argsort(b)[:k], np.argsort(p)[:k], np.argsort(y)[:k]
        vals = [float((y[f] > tau).mean()), float((y[a] > tau).mean()), float((y[o] > tau).mean())]
        out["coverage"].append(q)
        for key, v in zip(["fixed_bad", "adaptive_bad", "oracle_bad"], vals):
            out[key].append(v)
        out["fixed_mean"].append(float(y[f].mean()))
        out["adaptive_mean"].append(float(y[a].mean()))
        print(f"{q:8.0%} {vals[0]:10.1%} {vals[1]:13.1%} {vals[2]:11.1%} {y[f].mean():11.3f} {y[a].mean():11.3f}")
    # Within one photo: for each source, allow the same fraction of *its* target poses.
    keys = np.array([f"{r['scene']}/{r['src']}" for r in rows])
    print("within-photo (same image, which directions are safe):")
    out["within_photo"] = {"coverage": [], "fixed_bad": [], "adaptive_bad": []}
    for q in [0.25, 0.5, 0.75]:
        fb, ab, n = 0, 0, 0
        for key in np.unique(keys):
            idx = np.where(keys == key)[0]
            k = max(1, int(round(q * len(idx))))
            fb += (y[idx[np.argsort(b[idx])[:k]]] > tau).sum()
            ab += (y[idx[np.argsort(p[idx])[:k]]] > tau).sum()
            n += k
        out["within_photo"]["coverage"].append(q)
        out["within_photo"]["fixed_bad"].append(float(fb / n))
        out["within_photo"]["adaptive_bad"].append(float(ab / n))
        print(f"{q:8.0%} fixed bad {fb / n:6.1%}   adaptive bad {ab / n:6.1%}")
    return out


def safe_radius(rows, y, p, tau, ref=0.15, min_targets=8):
    """Per photo: fit LPIPS = a + b*log(offset) to measured and to LOSO-predicted values.

    radius = offset where the fit crosses tau (clipped to [0.005, 2]); also each fit's value at
    a reference offset. Baseline: one constant radius for every photo, the median measured
    radius over *other* scenes.
    """
    keys = np.array([f"{r['scene']}/{r['src']}" for r in rows])
    lb = np.log([r["baseline_rel"] for r in rows])
    photos = []
    for key in np.unique(keys):
        m = keys == key
        if m.sum() < min_targets:
            continue
        A = np.stack([np.ones(m.sum()), lb[m]], 1)
        (am, bm), (ap, bp) = np.linalg.lstsq(A, y[m], rcond=None)[0], np.linalg.lstsq(A, p[m], rcond=None)[0]
        rad = lambda a, b: float(np.clip(np.exp((tau - a) / b), 0.005, 2.0)) if b > 1e-4 else 2.0
        photos.append({"photo": key, "scene": key.split("/")[0], "r_meas": rad(am, bm), "r_pred": rad(ap, bp),
                       "at_ref_meas": float(am + bm * np.log(ref)), "at_ref_pred": float(ap + bp * np.log(ref))})
    rm = np.array([ph["r_meas"] for ph in photos])
    rp = np.array([ph["r_pred"] for ph in photos])
    sc = np.array([ph["scene"] for ph in photos])
    rc = np.array([np.median(rm[sc != s]) for s in sc])
    out = {
        "photos": len(photos), "tau": tau, "ref_offset": ref,
        "radius_spearman": spearman(rp, rm),
        "radius_log_err_pred": float(np.median(np.abs(np.log(rp / rm)))),
        "radius_log_err_const": float(np.median(np.abs(np.log(rc / rm)))),
        "at_ref_spearman": spearman(np.array([ph["at_ref_pred"] for ph in photos]), np.array([ph["at_ref_meas"] for ph in photos])),
        "per_photo": photos,
    }
    print(f"\nsafe radius over {len(photos)} photos: spearman {out['radius_spearman']:.3f}; median |log err| "
          f"predicted {out['radius_log_err_pred']:.3f} vs constant {out['radius_log_err_const']:.3f}; "
          f"LPIPS@{ref} spearman {out['at_ref_spearman']:.3f}")
    return out


def transfer(rows, lam=1.0):
    """Train on one dataset, test on the other (no scene or capture style shared)."""
    def fit(tr, te, cols):
        X = lambda R: np.array([[r[c] for c in cols] for r in R], float)
        Xtr, Xte = X(tr), X(te)
        ytr, yte = np.array([r["lpips"] for r in tr]), np.array([r["lpips"] for r in te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = np.hstack([(Xtr - mu) / sd, np.ones((len(tr), 1))])
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
        pr = np.hstack([(Xte - mu) / sd, np.ones((len(te), 1))]) @ w
        return float(1 - ((yte - pr) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()), spearman(pr, yte)
    llff = [r for r in rows if not r["scene"].startswith("m360_")]
    m360 = [r for r in rows if r["scene"].startswith("m360_")]
    out = {}
    for name, tr, te in [("llff_to_m360", llff, m360), ("m360_to_llff", m360, llff)]:
        out[name] = {}
        for label, cols in [("camera move only", MOVE), ("+ predicted disocclusion", MOVE + GEOM)]:
            r2, rho = fit(tr, te, cols)
            out[name][label] = {"r2": r2, "spearman": rho}
            print(f"transfer {name} {label}: R2 {r2:.3f} rho {rho:.3f}")
    return out


def main():
    import sys

    method = sys.argv[1] if len(sys.argv) > 1 else "sharp"
    rows = load(method)
    res = {"all": report(rows)}
    res["policies"] = policies(rows)
    y, p = loso(rows, MOVE + GEOM)
    res["transfer"] = transfer(rows)
    res["safe_radius"] = safe_radius(rows, y, p, res["policies"]["tau"])
    res["within_photo_rho"] = {}
    for ds in ["llff", "m360"]:
        idx = np.array([i for i, r in enumerate(rows) if r["scene"].startswith("m360_") == (ds == "m360")])
        keys = np.array([f"{rows[i]['scene']}/{rows[i]['src']}" for i in idx])
        bb = np.array([rows[i]["baseline_rel"] for i in idx])
        wp, wb = [], []
        for key in np.unique(keys):
            m = keys == key
            if m.sum() > 5:
                wp.append(spearman(p[idx][m], y[idx][m]))
                wb.append(spearman(bb[m], y[idx][m]))
        res["within_photo_rho"][ds] = {"predicted": float(np.mean(wp)), "distance": float(np.mean(wb)), "photos": len(wp)}
        print(f"within-photo rho {ds}: predicted {np.mean(wp):.3f} distance {np.mean(wb):.3f} ({len(wp)} photos)")
    res["scatter"] = [{"scene": r["scene"], "baseline_rel": r["baseline_rel"], "lpips": float(a), "pred": float(b)}
                      for r, a, b in zip(rows, y, p)]
    for ds, keep in {"llff": lambda r: not r["scene"].startswith("m360_"), "m360": lambda r: r["scene"].startswith("m360_")}.items():
        sub = [r for r in rows if keep(r)]
        if len({r["scene"] for r in sub}) >= 3:
            print(f"\n-- {ds}")
            res[ds] = report(sub)
    (ROOT / ("results/predictability.json" if method == "sharp" else f"results/predictability_{method}.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
