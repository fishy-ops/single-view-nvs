"""Fair evaluation of the background layer: composite it over SHARP, then score (Mac).

The training-time eval compared the layer *alone* (black where it predicts nothing) against SHARP's
full render, which is not how the method is used. Here the layer is shown only where it is opaque
inside the unseen mask; everywhere else SHARP's own pixels stand. Reported on held-out scenes:
mean L1 in the unseen region for SHARP, for the composite, and for a Telea fill of the same region.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from nvs.splat_torch import render_torch
from nvs.train_bglayer import BGNet, PAIRS, ROOT, features, layer_to_points, load_pair
from nvs.train_confidence import folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    import lpips as _lpips

    lp = _lpips.LPIPS(net="alex", spatial=True, verbose=False).to(dev)
    model = BGNet().to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt or ROOT / f"results/bglayer/fold{args.fold}.pt", map_location=dev))

    files = sorted(PAIRS.glob("*/*.npz"))
    fold_of = folds([{"scene": s} for s in sorted({f.parent.name for f in files})])
    test = [f for f in files if fold_of[f.parent.name] == args.fold]
    test = test[:: max(1, len(test) // args.limit)]

    rows = []
    with torch.no_grad():
        for f in test:
            p = load_pair(f, args.width)
            m = p["unseen"]
            if m.sum() < 100:
                continue
            out = model(features(p)[None].to(dev))[0]
            xyz, quats, radius, op, col = layer_to_points(out, p, dev)
            h, w = p["hw"]
            rgb, _, alpha = render_torch(xyz, quats, radius, op, col, p["w2c"].to(dev), p["K_tgt"].to(dev), w, h)
            gt = p["tgt_gt"].permute(1, 2, 0).to(dev)
            sharp = p["tgt_render"].permute(1, 2, 0).to(dev)
            use = m.to(dev) & (alpha > 0.5)
            comp = torch.where(use[..., None], rgb.clamp(0, 1), sharp)
            # "holes only": trust SHARP wherever it has real content; use the layer only where it is
            # empty (out of frame, or fully transparent), which the diagnostic says is where it helps.
            empty = torch.from_numpy(np.load(f)["tgt_alpha"]).float().to(dev)
            empty = torch.nn.functional.interpolate(empty[None, None] / 255, size=alpha.shape, mode="bilinear")[0, 0] < 0.5
            use_holes = empty & (alpha > 0.5)
            comp_holes = torch.where(use_holes[..., None], rgb.clamp(0, 1), sharp)
            img = (sharp.cpu().numpy() * 255).astype(np.uint8)
            holes = (m.numpy() & (p["src_alpha"].numpy() < 0.5)).astype(np.uint8)
            telea = torch.from_numpy(cv2.inpaint(img, holes, 5, cv2.INPAINT_TELEA) / 255).float().to(dev) if holes.any() else sharp
            mm = m.to(dev)
            l1 = lambda x: float((x - gt).abs().mean(-1)[mm].mean())
            # LPIPS restricted to the unseen region: L1 rewards SHARP's smear, as PSNR did earlier
            lpx = lambda x: float(lp(x.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1)[0, 0][mm].mean())
            # where the layer actually paints: is it better or worse than SHARP's smear there?
            u = use
            in_used = lambda x: float((x - gt).abs().mean(-1)[u].mean()) if u.any() else float("nan")
            rows.append({"scene": p["meta"]["scene"], "src": p["meta"]["src"], "tgt": p["meta"]["tgt"],
                         "l1_sharp": l1(sharp), "l1_comp": l1(comp), "l1_telea": l1(telea),
                         "l1_layer_only": l1(rgb.clamp(0, 1)), "cover": float(use.float().sum() / mm.float().sum()),
                         "used_layer": in_used(rgb.clamp(0, 1)), "used_sharp": in_used(sharp),
                         "l1_holes_only": l1(comp_holes), "hole_share": float(use_holes.float().mean()),
                         "lpips_sharp": lpx(sharp), "lpips_comp": lpx(comp), "lpips_holes": lpx(comp_holes)})
    a = lambda k: float(np.mean([r[k] for r in rows]))
    summ = {"n": len(rows), "fold": args.fold,
            "l1_sharp": a("l1_sharp"), "l1_composite": a("l1_comp"), "l1_telea": a("l1_telea"),
            "l1_layer_only": a("l1_layer_only"), "mean_cover": a("cover"),
            "composite_beats_sharp": float(np.mean([r["l1_comp"] < r["l1_sharp"] for r in rows])),
            "used_layer": float(np.nanmean([r["used_layer"] for r in rows])),
            "used_sharp": float(np.nanmean([r["used_sharp"] for r in rows])),
            "layer_beats_sharp_where_used": float(np.nanmean([r["used_layer"] < r["used_sharp"] for r in rows])),
            "l1_holes_only": float(np.mean([r["l1_holes_only"] for r in rows])),
            "holes_only_beats_sharp": float(np.mean([r["l1_holes_only"] < r["l1_sharp"] for r in rows])),
            "hole_share": float(np.mean([r["hole_share"] for r in rows])),
            "lpips_sharp": float(np.mean([r["lpips_sharp"] for r in rows])),
            "lpips_composite": float(np.mean([r["lpips_comp"] for r in rows])),
            "lpips_holes_only": float(np.mean([r["lpips_holes"] for r in rows])),
            "lpips_composite_beats_sharp": float(np.mean([r["lpips_comp"] < r["lpips_sharp"] for r in rows]))}
    run_dir = Path(args.ckpt).parent if args.ckpt else ROOT / "results/bglayer"
    (ROOT / run_dir / f"fair_eval_fold{args.fold}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print("FAIR", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
