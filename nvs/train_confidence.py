"""Per-pixel confidence for single-image view synthesis (Mac / MPS).

Given only test-time signals at a requested camera pose (SHARP render RGB, alpha, relative depth,
SHARP-predicted "unseen by the source photo" mask), predict the per-pixel LPIPS error map against
the real photo. Scene-level 3-fold cross-validation (no scene appears in both train and test).

Compared against a "mask prior" (least squares on unseen + (1 - alpha)), which is what you get
for free from SHARP's geometry. Results go to results/confidence/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "confidence"


def load_pairs():
    X, Y, meta = [], [], []
    for f in sorted((ROOT / "results" / "pairs").glob("*/*.npz")):
        z = np.load(f)
        x = np.concatenate(
            [
                z["render"].astype(np.float16) / 255,
                (z["alpha"].astype(np.float16) / 255)[..., None],
                np.clip(z["depth_rel"], -3, 3)[..., None].astype(np.float16) / 3,
                (z["unseen"].astype(np.float16) / 255)[..., None],
            ],
            -1,
        )
        X.append(torch.from_numpy(x).permute(2, 0, 1))
        Y.append(torch.from_numpy(z["lpips"].astype(np.float16))[None])
        meta.append(json.loads(str(z["meta"])))
    return X, Y, meta


def folds(meta, k=3):
    """Deal scenes round-robin into k folds, separately for LLFF and 360, so each fold has both."""
    scenes = sorted({m["scene"] for m in meta})
    llff = [s for s in scenes if not s.startswith("m360_")]
    m360 = [s for s in scenes if s.startswith("m360_")]
    fold_of = {}
    for group in (llff, m360):
        for i, s in enumerate(group):
            fold_of[s] = i % k
    return fold_of


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin=6, ch=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        c = cin
        for co in ch:
            self.downs.append(Block(c, co))
            c = co
        self.ups = nn.ModuleList()
        for co in reversed(ch[:-1]):
            self.ups.append(Block(c + co, co))
            c = co
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x if i == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        for u, s in zip(self.ups, reversed(skips[:-1])):
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = u(torch.cat([x, s], 1))
        return F.softplus(self.head(x))


def rank(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def auroc(score, label):
    r = rank(score)
    pos = label.sum()
    neg = len(label) - pos
    return float((r[label].sum() - pos * (pos - 1) / 2) / (pos * neg)) if pos and neg else float("nan")


def batch(X, Y, idx, crop, dev, train=True):
    xs, ys = [], []
    for i in idx:
        x, y = X[i], Y[i]
        if train:
            _, h, w = x.shape
            t, l = np.random.randint(0, h - crop + 1), np.random.randint(0, w - crop + 1)
            x, y = x[:, t:t + crop, l:l + crop], y[:, t:t + crop, l:l + crop]
            if np.random.rand() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs).float().to(dev), torch.stack(ys).float().to(dev)


def evaluate(pred_fn, X, Y, idx, meta, thr, rng):
    pix_p, pix_y, img_p, img_y = [], [], [], []
    for i in idx:
        p = pred_fn(i)
        y = Y[i][0].float().numpy()
        img_p.append(p.mean())
        img_y.append(y.mean())
        sel = rng.choice(p.size, size=min(4000, p.size), replace=False)
        pix_p.append(p.reshape(-1)[sel])
        pix_y.append(y.reshape(-1)[sel])
    pp, py = np.concatenate(pix_p), np.concatenate(pix_y)
    return {
        "pixel_spearman": spearman(pp, py),
        "pixel_auroc_bad": auroc(pp, py > thr),
        "pixel_mae": float(np.abs(pp - py).mean()),
        "image_spearman": spearman(np.array(img_p), np.array(img_y)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=240)  # 360 pairs are 384x255
    ap.add_argument("--out", default="confidence", help="results subdirectory")
    args = ap.parse_args()
    global OUT
    OUT = ROOT / "results" / args.out
    torch.set_num_threads(4)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    X, Y, meta = load_pairs()
    fold_of = folds(meta)
    print(f"loaded {len(X)} pairs from {len(fold_of)} scenes in {time.time() - t0:.0f}s", flush=True)

    results = []
    for k in range(3):
        rng = np.random.default_rng(0)
        tr = [i for i, m in enumerate(meta) if fold_of[m["scene"]] != k]
        te = [i for i, m in enumerate(meta) if fold_of[m["scene"]] == k]
        thr = float(np.quantile(np.concatenate([Y[i].float().numpy().reshape(-1)[::97] for i in tr]), 0.8))

        # Mask prior: y ~ a + b*unseen + c*(1-alpha), least squares on subsampled train pixels.
        A, b = [], []
        for i in tr:
            x = X[i].float().numpy()
            A.append(np.stack([np.ones(x[0].size), x[5].reshape(-1), 1 - x[3].reshape(-1)], 1)[::53])
            b.append(Y[i].float().numpy().reshape(-1)[::53])
        coef = np.linalg.lstsq(np.concatenate(A), np.concatenate(b), rcond=None)[0]
        prior = lambda i: (coef[0] + coef[1] * X[i][5].float().numpy() + coef[2] * (1 - X[i][3].float().numpy()))
        const = lambda i: np.full(Y[i][0].shape, float(np.mean(np.concatenate(b))), np.float32)

        model = UNet().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        steps = args.epochs * max(1, len(tr) // args.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
        step = 0
        for ep in range(args.epochs):
            model.train()
            perm = np.random.permutation(tr)
            losses = []
            for j in range(0, len(perm) - args.bs + 1, args.bs):
                x, y = batch(X, Y, perm[j:j + args.bs], args.crop, dev)
                loss = F.l1_loss(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                step += 1
                losses.append(loss.item())
            if ep % 10 == 9 or ep == args.epochs - 1:
                print(f"fold {k} epoch {ep + 1} train_l1 {np.mean(losses):.4f} ({time.time() - t0:.0f}s)", flush=True)

        model.eval()

        @torch.no_grad()
        def net(i):
            return model(X[i][None].float().to(dev))[0, 0].cpu().numpy()

        res = {
            "fold": k,
            "test_scenes": sorted({meta[i]["scene"] for i in te}),
            "n_train": len(tr),
            "n_test": len(te),
            "bad_threshold": thr,
            "prior_coef": coef.tolist(),
            "constant": evaluate(const, X, Y, te, meta, thr, rng),
            "mask_prior": evaluate(prior, X, Y, te, meta, thr, rng),
            "unet": evaluate(net, X, Y, te, meta, thr, rng),
        }
        results.append(res)
        print(json.dumps({m: res[m] for m in ["test_scenes", "mask_prior", "unet"]}), flush=True)
        torch.save(model.state_dict(), OUT / f"unet_fold{k}.pt")

        # Figures: render | GT | actual LPIPS | predicted, for the highest-baseline test pair per scene.
        for s in res["test_scenes"]:
            i = max((i for i in te if meta[i]["scene"] == s), key=lambda i: meta[i]["baseline_rel"])
            x = X[i].float().numpy()
            heat = lambda m: (np.clip(m / 0.8, 0, 1)[..., None] * np.array([235, 64, 52]) + (1 - np.clip(m / 0.8, 0, 1))[..., None] * 20).astype(np.uint8)
            strip = np.concatenate([
                (x[:3].transpose(1, 2, 0) * 255).astype(np.uint8),
                np.load(sorted((ROOT / "results/pairs" / s).glob(f"s{meta[i]['src']:03d}_t{meta[i]['tgt']:03d}.npz"))[0])["gt"],
                heat(Y[i][0].float().numpy()),
                heat(net(i)),
            ], 1)
            Image.fromarray(strip).save(OUT / f"fig_{s}.jpg", quality=88)

    summary = {m: {key: float(np.mean([r[m][key] for r in results])) for key in results[0][m]}
               for m in ["constant", "mask_prior", "unet"]}
    (OUT / "results.json").write_text(json.dumps({"folds": results, "mean": summary}, indent=1))
    print("MEAN", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
