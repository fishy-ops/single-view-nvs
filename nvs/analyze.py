"""Summarise results/eval/<method>/*.jsonl: overall, by baseline bin, by scene, paired gains."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["psnr", "ssim", "lpips", "psnr_covis", "lpips_covis", "psnr_disocc", "lpips_disocc"]


def load(method: str) -> dict[tuple, dict]:
    rows = {}
    for f in sorted((ROOT / "results" / "eval" / method).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            rows[(r["scene"], r["src"], r["tgt"])] = r
    return rows


def mean(rows, key) -> float:
    v = np.array([r[key] for r in rows], dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def table(title, header, body):
    print(f"\n== {title}")
    cells = [header] + [[c if isinstance(c, str) else f"{c:.3f}" for c in row] for row in body]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    for r in cells:
        print("  ".join(c.rjust(w) for c, w in zip(r, widths)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*")
    args = ap.parse_args()
    methods = args.methods or sorted(p.name for p in (ROOT / "results" / "eval").iterdir() if p.is_dir())
    data = {m: load(m) for m in methods}
    ref = "sharp" if "sharp" in data else methods[0]
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    print(f"joined rows across {methods}: {len(keys)} (of {[len(d) for d in data.values()]})")
    joined = {m: [data[m][k] for k in keys] for m in methods}
    joined["identity"] = [dict(data[ref][k], psnr=data[ref][k]["id_psnr"], lpips=data[ref][k]["id_lpips"],
                               **{x: math.nan for x in METRICS if x not in ("psnr", "lpips")}) for k in keys]
    out = {"methods": methods, "n_rows": len(keys)}

    names = methods + ["identity"]
    out["overall"] = {m: {x: mean(joined[m], x) for x in METRICS} for m in names}
    table("overall", ["method", "n"] + METRICS, [[m, float(len(keys))] + [out["overall"][m][x] for x in METRICS] for m in names])

    b = np.array([data[ref][k]["baseline_rel"] for k in keys])
    edges = np.quantile(b, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    idx = np.clip(np.searchsorted(edges, b, side="right") - 1, 0, 4)
    out["bin_edges"] = edges.tolist()
    out["by_bin"] = []
    body = []
    for i in range(5):
        sel = np.where(idx == i)[0]
        entry = {"lo": edges[i], "hi": edges[i + 1], "n": int(len(sel)),
                 "covis_frac": mean([joined[ref][j] for j in sel], "covis_frac"),
                 "rot_deg": mean([joined[ref][j] for j in sel], "rot_deg")}
        for m in names:
            entry[m] = {x: mean([joined[m][j] for j in sel], x) for x in ["psnr", "lpips", "lpips_covis", "lpips_disocc"]}
        out["by_bin"].append(entry)
        body.append([f"{edges[i]:.3f}-{edges[i+1]:.3f}", entry["covis_frac"], entry["rot_deg"]]
                    + [entry[m][x] for m in names for x in ["psnr", "lpips"]])
    table("by baseline bin (baseline / median source depth)", ["bin", "covis", "rot"] + [f"{m[:10]}:{x}" for m in names for x in ["psnr", "lpips"]], body)

    scenes = sorted({k[0] for k in keys})
    out["by_scene"] = {}
    body = []
    for s in scenes:
        sel = [j for j, k in enumerate(keys) if k[0] == s]
        e = {m: {x: mean([joined[m][j] for j in sel], x) for x in ["psnr", "lpips"]} for m in names}
        e["align_err"] = mean([joined[ref][j] for j in sel], "align_err")
        out["by_scene"][s] = e
        body.append([s, e["align_err"]] + [e[m][x] for m in names for x in ["psnr", "lpips"]])
    table("by scene", ["scene", "align"] + [f"{m[:10]}:{x}" for m in names for x in ["psnr", "lpips"]], body)

    out["paired_vs_" + ref] = {}
    body = []
    for m in names:
        if m == ref:
            continue
        a, o = joined[ref], joined[m]
        dp = float(np.nanmean([x["psnr"] - y["psnr"] for x, y in zip(a, o)]))
        dl = float(np.nanmean([y["lpips"] - x["lpips"] for x, y in zip(a, o)]))
        win = float(np.mean([x["lpips"] < y["lpips"] for x, y in zip(a, o)]))
        out["paired_vs_" + ref][m] = {"d_psnr": dp, "d_lpips": dl, "lpips_win_rate": win}
        body.append([m, dp, dl, win])
    table(f"paired: {ref} minus other (positive = {ref} better)", ["other", "dPSNR", "dLPIPS", "win"], body)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return None if isinstance(o, float) and not math.isfinite(o) else o

    (ROOT / "results" / "summary.json").write_text(json.dumps(clean(out), indent=1))


if __name__ == "__main__":
    main()
