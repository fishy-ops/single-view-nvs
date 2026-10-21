"""Apple Depth Pro depth for each SHARP source view (Mac / MPS). Scale is re-fit at eval time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast

from nvs.scenes import M360, SCENES, load_scene

ROOT = Path(__file__).resolve().parents[1]


def main():
    dev = torch.device("mps")
    proc = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(dev).eval()
    for name in SCENES + M360:
        scene = load_scene(name)
        for src_file in sorted((ROOT / "results" / "sharp" / name).glob("src*.pt")):
            si = int(src_file.stem[3:])
            out = ROOT / "results" / "depth" / "depthpro" / name / f"src{si:03d}.npy"
            if out.exists():
                continue
            img = Image.open(scene.views[si].path).convert("RGB")
            inputs = proc(images=img, return_tensors="pt").to(dev)
            with torch.no_grad():
                # HF DepthPro's raw predicted_depth is canonical *inverse* depth (log-corr with
                # COLMAP z is -0.9 on trex/room); scale is re-fit at eval, so 1/x is enough.
                d = 1.0 / model(**inputs).predicted_depth.clamp(min=1e-4)  # (1, H', W')
            d = F.interpolate(d[:, None], size=(scene.height, scene.width), mode="bilinear", align_corners=False)[0, 0]
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, d.float().cpu().numpy())
            print("depthpro", name, si, float(d.min()), float(d.max()), flush=True)


if __name__ == "__main__":
    main()
