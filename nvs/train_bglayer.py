"""Train a background-layer head on top of frozen SHARP (Mac / MPS).

Idea: SHARP is good at what the photo saw and empty behind it. Keep SHARP untouched and learn one
extra layer: for every source pixel, the colour and depth of the surface *hidden behind* it. That
layer is 3D, so it is rendered with the scene and cannot flicker when the camera moves.

Input (all from the source view, no ground truth): source RGB, log depth relative to the median,
alpha, and a depth-edge map. Output per pixel: background colour, how far behind the front surface
it sits, and an opacity.

Training: unproject the predicted layer into the source camera frame, render it at the target
camera with nvs/splat_torch.py, and compare with the real photo **only where the source photo saw
nothing**. Scenes are split into folds, so every reported number comes from unseen scenes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nvs.splat_torch import render_torch
from nvs.train_confidence import UNet, folds


class BGNet(UNet):
    """UNet with a raw 5-channel head: softplus in the parent would clamp colours and logits."""

    def __init__(self):
        super().__init__(cin=9)
        self.head = torch.nn.Conv2d(32, 5, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x if i == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        for u, s in zip(self.ups, reversed(skips[:-1])):
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = u(torch.cat([x, s], 1))
        return self.head(x)

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "results_local" / "train_pairs"


def load_pair(path, width):
    z = np.load(path)
    m = json.loads(str(z["meta"]))
    h0, w0 = z["src_rgb"].shape[:2]
    s = width / w0
    h = int(round(h0 * s)) // 8 * 8
    rs = lambda a, mode="bilinear": F.interpolate(torch.from_numpy(np.ascontiguousarray(a)).float()[None, None], size=(h, width), mode=mode)[0, 0]
    rgb = lambda a: F.interpolate(torch.from_numpy(np.ascontiguousarray(a)).float().permute(2, 0, 1)[None] / 255, size=(h, width), mode="bilinear")[0]
    K = z["K"].copy()
    K[:2] *= s
    Kt = z["K_tgt"].copy()
    Kt[:2] *= s
    out = {
        "src_rgb": rgb(z["src_rgb"]), "src_depth": rs(z["src_depth"].astype(np.float32)),
        "src_alpha": rs(z["src_alpha"].astype(np.float32) / 255),
        "tgt_gt": rgb(z["tgt_gt"]), "tgt_render": rgb(z["tgt_render"]),
        "unseen": rs(z["unseen"].astype(np.float32), "nearest") > 0.5,
        "K": torch.from_numpy(K).float(), "K_tgt": torch.from_numpy(Kt).float(),
        "w2c": torch.from_numpy(z["w2c"]).float(), "meta": m, "hw": (h, width),
    }
    out["bg_init"] = background_estimate(out)
    return out


def background_estimate(p):
    """Classical guess at what sits behind each foreground edge: inpaint the source view where the
    depth jumps, so the model only has to predict a *correction* rather than invent colour."""
    import cv2

    d = p["src_depth"].numpy()
    logd = np.log(np.clip(d, 1e-4, None))
    gx = np.abs(np.diff(logd, axis=1, prepend=logd[:, :1]))
    gy = np.abs(np.diff(logd, axis=0, prepend=logd[:1]))
    edge = ((gx + gy) > 0.05).astype(np.uint8)
    fg = cv2.dilate(edge, np.ones((7, 7), np.uint8))  # a band around every depth discontinuity
    img = (p["src_rgb"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    filled = cv2.inpaint(img, fg, 7, cv2.INPAINT_TELEA)
    return torch.from_numpy(filled).float().permute(2, 0, 1) / 255


def features(p):
    d = p["src_depth"].clamp(min=1e-4)
    logd = torch.log(d / d.median())
    gx = F.pad((logd[:, 1:] - logd[:, :-1]).abs(), (1, 0))
    gy = F.pad((logd[1:] - logd[:-1]).abs(), (0, 0, 1, 0))
    edge = (gx + gy).clamp(max=1.0)
    return torch.cat([p["src_rgb"], logd.clamp(-3, 3)[None] / 3, p["src_alpha"][None], edge[None],
                      p["bg_init"]], 0)  # 9 channels: + the classical background estimate


def layer_to_points(out, p, dev):
    """Head output -> 3D points in the source camera frame (colour, depth behind the front surface)."""
    h, w = p["hw"]
    # residual over the classical estimate, so "do nothing" is already a sensible answer
    base = p["bg_init"].to(dev)
    col = (base + torch.tanh(out[:3]) * 0.5).clamp(0, 1).permute(1, 2, 0).reshape(-1, 3)
    behind = 1.0 + F.softplus(out[3])  # strictly behind the visible surface
    op = torch.sigmoid(out[4]).reshape(-1)
    depth = (p["src_depth"].to(dev).clamp(min=1e-4) * behind).reshape(-1)
    ys, xs = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    pix = torch.stack([xs.reshape(-1) + 0.5, ys.reshape(-1) + 0.5, torch.ones(h * w, device=dev)], -1)
    xyz = (pix @ torch.linalg.inv(p["K"].to(dev)).T) * depth[:, None]
    radius = (depth / p["K"][0, 0].to(dev) * 0.9)[:, None].expand(-1, 3)
    quats = torch.tensor([1.0, 0, 0, 0], device=dev).expand(h * w, 4)
    return xyz, quats, radius, op, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="bglayer")
    ap.add_argument("--crop", type=int, default=128, help="render only this crop per step (memory)")
    ap.add_argument("--lpips-every", type=int, default=4, help="perceptual term every N steps (memory)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-train", type=int, default=0, help="cap the training set (data-scaling ablation)")
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = ROOT / "results" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(PAIRS.glob("*/*.npz"))
    scenes = sorted({f.parent.name for f in files})
    fold_of = folds([{"scene": s} for s in scenes])
    train = [f for f in files if fold_of[f.parent.name] != args.fold]
    test = [f for f in files if fold_of[f.parent.name] == args.fold]
    if args.max_train and len(train) > args.max_train:
        step_ = len(train) / args.max_train  # keep the scene mix, just thin it out
        train = [train[int(i * step_)] for i in range(args.max_train)]
    print(f"{len(train)} train pairs, {len(test)} test pairs, test scenes "
          f"{sorted({f.parent.name for f in test})}", flush=True)

    model = BGNet().to(dev)  # colour(3) + depth-behind(1) + opacity(1)
    ckpt = out_dir / f"fold{args.fold}.pt"
    if args.resume and ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        print(f"resumed from {ckpt}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    try:
        import lpips as _lpips

        lp_fn = _lpips.LPIPS(net="alex", verbose=False).to(dev)
        for q in lp_fn.parameters():
            q.requires_grad_(False)
    except Exception as e:
        print("no LPIPS term:", type(e).__name__, flush=True)
        lp_fn = None
    t0 = time.time()
    hist = []

    def forward(p):
        return model(features(p)[None].to(dev))[0]

    for step in range(1, args.steps + 1):
        p = load_pair(train[rng.integers(len(train))], args.width)
        out = forward(p)
        xyz, quats, radius, op, col = layer_to_points(out, p, dev)
        h, w = p["hw"]
        cw = min(args.crop, w, h)
        um = p["unseen"].numpy()
        ys_u, xs_u = np.nonzero(um)
        if len(ys_u) < 100:
            continue
        j = rng.integers(len(ys_u))  # centre the crop on an unseen pixel
        x0 = int(np.clip(xs_u[j] - cw // 2, 0, w - cw))
        y0 = int(np.clip(ys_u[j] - cw // 2, 0, h - cw))
        rgb, _, alpha = render_torch(xyz, quats, radius, op, col, p["w2c"].to(dev), p["K_tgt"].to(dev), w, h,
                                     window=(x0, y0, cw, cw))
        gt = p["tgt_gt"].permute(1, 2, 0)[y0:y0 + cw, x0:x0 + cw].to(dev)
        m = p["unseen"][y0:y0 + cw, x0:x0 + cw].to(dev)
        if m.sum() < 100:
            continue
        loss_rgb = (rgb - gt).abs().mean(-1)[m].mean()
        loss_cover = (1 - alpha[m]).mean()  # the layer should actually cover the unseen pixels
        # perceptual term on the crop: L1 alone rewards blur, and SHARP's smear already beats blur
        masked = lambda t: (t.permute(2, 0, 1)[None] * m[None, None] + 0.5 * (~m)[None, None]) * 2 - 1
        use_lp = lp_fn is not None and step % args.lpips_every == 0
        loss_lp = lp_fn(masked(rgb), masked(gt)).mean() if use_lp else torch.zeros((), device=dev)
        loss = loss_rgb + 0.1 * loss_cover + (0.3 * args.lpips_every) * loss_lp
        try:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        except RuntimeError as e:  # transient allocator pressure: skip the step rather than die
            print(f"step {step} skipped: {type(e).__name__}", flush=True)
            opt.zero_grad(set_to_none=True)
            if dev.type == "mps":
                torch.mps.empty_cache()
            sched.step()
            continue
        sched.step()
        hist.append(float(loss_rgb))
        hist[:] = hist[-500:]
        if step % 20 == 0 and dev.type == "mps":
            torch.mps.empty_cache()
        if step % 100 == 0:
            print(f"step {step} loss_rgb {np.mean(hist[-100:]):.4f} cover {float(loss_cover):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step % 1000 == 0 or step == args.steps:
            torch.save(model.state_dict(), out_dir / f"fold{args.fold}.pt")

    # Held-out scenes: L1 in the unseen region, against SHARP's own render as the baseline.
    model.eval()
    rows = []
    with torch.no_grad():
        for f in test[:: max(1, len(test) // 120)]:
            p = load_pair(f, args.width)
            out = forward(p)
            xyz, quats, radius, op, col = layer_to_points(out, p, dev)
            h, w = p["hw"]
            rgb, _, _ = render_torch(xyz, quats, radius, op, col, p["w2c"].to(dev), p["K_tgt"].to(dev), w, h)
            gt = p["tgt_gt"].permute(1, 2, 0).to(dev)
            base = p["tgt_render"].permute(1, 2, 0).to(dev)
            m = p["unseen"].to(dev)
            if m.sum() < 100:
                continue
            rows.append({"scene": p["meta"]["scene"], "src": p["meta"]["src"], "tgt": p["meta"]["tgt"],
                         "l1_model": float((rgb - gt).abs().mean(-1)[m].mean()),
                         "l1_sharp": float((base - gt).abs().mean(-1)[m].mean())})
    summ = {"fold": args.fold, "n": len(rows), "steps": args.steps, "width": args.width,
            "l1_model": float(np.mean([r["l1_model"] for r in rows])),
            "l1_sharp": float(np.mean([r["l1_sharp"] for r in rows])),
            "win_rate": float(np.mean([r["l1_model"] < r["l1_sharp"] for r in rows]))}
    (out_dir / f"eval_fold{args.fold}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print("EVAL", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
