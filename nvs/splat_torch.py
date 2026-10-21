"""Differentiable Gaussian splatting in plain PyTorch (runs on MPS, CPU or CUDA).

gsplat's rasteriser is CUDA-only, so training on the Mac needs its own renderer. This is a
tile-free soft-z-buffer splatter: each Gaussian is projected to a 2D ellipse (EWA style), sorted by
depth, and composited front-to-back with alpha. It trades speed for portability and gradients.

Verify against gsplat with `nvs/check_splat.py` (renders the same scene both ways on the desktop).
"""

from __future__ import annotations

import torch


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


def project(means, quats, scales, w2c, K, width, height, eps2d: float = 0.3):
    """Camera-space projection: returns 2D centres, 2x2 covariances, depths and a visibility mask."""
    R, t = w2c[:3, :3], w2c[:3, 3]
    cam = means @ R.T + t
    z = cam[:, 2]
    ok = z > 1e-3
    zc = z.clamp(min=1e-3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uv = torch.stack([fx * cam[:, 0] / zc + cx, fy * cam[:, 1] / zc + cy], -1)

    Rg = quat_to_rotmat(quats)
    S = torch.diag_embed(scales)
    cov3d = Rg @ S @ S.transpose(-1, -2) @ Rg.transpose(-1, -2)
    cov_cam = R @ cov3d @ R.T  # world -> camera

    # Jacobian of the perspective projection at each Gaussian centre.
    zeros = torch.zeros_like(zc)
    J = torch.stack([
        torch.stack([fx / zc, zeros, -fx * cam[:, 0] / zc**2], -1),
        torch.stack([zeros, fy / zc, -fy * cam[:, 1] / zc**2], -1),
    ], -2)
    cov2d = J @ cov_cam @ J.transpose(-1, -2)
    cov2d = cov2d + torch.eye(2, device=means.device) * eps2d  # low-pass, as in EWA splatting
    inside = (uv[:, 0] > -50) & (uv[:, 0] < width + 50) & (uv[:, 1] > -50) & (uv[:, 1] < height + 50)
    return uv, cov2d, z, ok & inside


def render_torch(means, quats, scales, opacities, colors, w2c, K, width, height,
                 tile: int = 32, chunk: int = 256, bg: float = 0.0, min_T: float = 0.003, window=None):
    """Render one view. Returns (H, W, 3) colour, (H, W) depth and (H, W) alpha.

    Per tile the Gaussians are composited front-to-back in depth-sorted chunks, carrying the running
    transmittance T, so peak memory is chunk x tile^2 rather than all splats at once. A tile stops
    early once every pixel in it is opaque (T < min_T), which is what makes this affordable.

    `window=(x0, y0, w, h)` renders only that crop (in full-image pixel coordinates) and returns
    crop-sized tensors. Training uses it to keep the autograd graph small.
    """
    dev = means.device
    uv, cov2d, z, vis = project(means, quats, scales, w2c, K, width, height)
    idx = torch.nonzero(vis, as_tuple=True)[0]
    if idx.numel() == 0:
        return (torch.full((height, width, 3), bg, device=dev), torch.zeros(height, width, device=dev),
                torch.zeros(height, width, device=dev))
    idx = idx[torch.argsort(z[idx])]
    uv, cov2d, z = uv[idx], cov2d[idx], z[idx]
    op, col = opacities[idx], colors[idx]

    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    conic = torch.stack([cov2d[:, 1, 1], -cov2d[:, 0, 1], cov2d[:, 0, 0]], -1) / det.clamp(min=1e-8)[:, None]
    radius = 3.0 * torch.sqrt(torch.stack([cov2d[:, 0, 0], cov2d[:, 1, 1]], -1).max(-1).values.clamp(min=1e-6))

    wx0, wy0, ww, wh = window if window is not None else (0, 0, width, height)
    out = torch.zeros(wh, ww, 3, device=dev)
    depth = torch.zeros(wh, ww, device=dev)
    alpha_acc = torch.zeros(wh, ww, device=dev)
    ys = torch.arange(height, device=dev).float() + 0.5
    xs = torch.arange(width, device=dev).float() + 0.5

    for y0 in range(wy0, wy0 + wh, tile):
        for x0 in range(wx0, wx0 + ww, tile):
            y1, x1 = min(y0 + tile, wy0 + wh), min(x0 + tile, wx0 + ww)
            oy, ox = y0 - wy0, x0 - wx0  # offsets into the crop-sized outputs
            in_tile = ((uv[:, 0] + radius >= x0) & (uv[:, 0] - radius < x1)
                       & (uv[:, 1] + radius >= y0) & (uv[:, 1] - radius < y1))
            sel = torch.nonzero(in_tile, as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            gxs = xs[x0:x1][None, None, :]
            gys = ys[y0:y1][None, :, None]
            T = torch.ones(y1 - y0, x1 - x0, device=dev)
            c_acc = torch.zeros(y1 - y0, x1 - x0, 3, device=dev)
            d_acc = torch.zeros(y1 - y0, x1 - x0, device=dev)
            a_acc = torch.zeros(y1 - y0, x1 - x0, device=dev)
            for c0 in range(0, sel.numel(), chunk):
                cs = sel[c0:c0 + chunk]
                gx = gxs - uv[cs, 0][:, None, None]
                gy = gys - uv[cs, 1][:, None, None]
                power = -0.5 * (conic[cs, 0][:, None, None] * gx**2
                                + 2 * conic[cs, 1][:, None, None] * gx * gy
                                + conic[cs, 2][:, None, None] * gy**2)
                a = (op[cs][:, None, None] * torch.exp(power.clamp(max=0.0))).clamp(max=0.999)
                one_minus = 1 - a
                trans_in = torch.cumprod(torch.cat([torch.ones_like(one_minus[:1]), one_minus[:-1]], 0), 0)
                w = a * trans_in * T
                c_acc = c_acc + (w[..., None] * col[cs][:, None, None, :]).sum(0)
                d_acc = d_acc + (w * z[cs][:, None, None]).sum(0)
                a_acc = a_acc + w.sum(0)
                T = T * one_minus.prod(0)
                if float(T.max()) < min_T:
                    break
            out[oy:oy + (y1 - y0), ox:ox + (x1 - x0)] = c_acc + bg * (1 - a_acc)[..., None]
            depth[oy:oy + (y1 - y0), ox:ox + (x1 - x0)] = d_acc
            alpha_acc[oy:oy + (y1 - y0), ox:ox + (x1 - x0)] = a_acc
    depth = depth / alpha_acc.clamp(min=1e-6)
    return out, depth, alpha_acc
