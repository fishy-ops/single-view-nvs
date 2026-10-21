"""Run SHARP on source views of every LLFF scene (Mac / MPS). Saves Gaussians as fp16 .pt."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sharp.cli.predict import DEFAULT_MODEL_URL, predict_image
from sharp.models import PredictorParams, create_predictor

from nvs.scenes import SCENES, load_scene

OUT = Path(__file__).resolve().parents[1] / "results" / "sharp"


def pick_sources(scene, n: int) -> list[int]:
    """Center view plus views at evenly spaced distance quantiles from it."""
    c = scene.center_view_index()
    if n == 1:
        return [c]
    centers = np.stack([v.center for v in scene.views])
    order = np.argsort(np.linalg.norm(centers - centers[c], axis=1))
    picks = [c] + [int(order[int(q * (len(order) - 1))]) for q in np.linspace(0.4, 0.9, n - 1)]
    return list(dict.fromkeys(picks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--sources", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = create_predictor(PredictorParams())
    model.load_state_dict(torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=False))
    model.eval().to(device)

    for name in args.scenes:
        scene = load_scene(name)
        for si in pick_sources(scene, args.sources):
            out = OUT / name / f"src{si:03d}.pt"
            if out.exists():
                continue
            v = scene.views[si]
            image = np.asarray(Image.open(v.path).convert("RGB"))
            f_px = float(v.K[0, 0])
            t0 = time.time()
            g = predict_image(model, image, f_px, device)
            torch.mps.synchronize() if device.type == "mps" else None
            dt = time.time() - t0
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "means": g.mean_vectors[0].half().cpu(),
                    "quats": g.quaternions[0].half().cpu(),
                    "scales": g.singular_values[0].half().cpu(),
                    "opacities": g.opacities[0].half().cpu(),
                    "colors": g.colors[0].half().cpu(),  # linear RGB
                    "f_px": f_px,
                    "source_index": si,
                    "seconds": dt,
                },
                out,
            )
            print(f"{name} src{si:03d}: {g.mean_vectors.shape[1]} gaussians, {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
