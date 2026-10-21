"""LLFF scene loading: COLMAP poses/intrinsics at images_4 resolution, sparse points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PIL import Image

DATA = Path(__file__).resolve().parents[1] / "data" / "llff"
SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]


@dataclass
class View:
    name: str  # original COLMAP image name
    path: Path  # images_4 png
    K: np.ndarray  # 3x3 at images_4 resolution
    w2c: np.ndarray  # 4x4 OpenCV world->camera
    point_ids: np.ndarray  # COLMAP 3D point ids observed in this view

    @property
    def center(self) -> np.ndarray:
        R, t = self.w2c[:3, :3], self.w2c[:3, 3]
        return -R.T @ t


@dataclass
class Scene:
    name: str
    views: list[View]
    points: dict[int, np.ndarray]  # point3D id -> xyz
    width: int
    height: int

    def center_view_index(self) -> int:
        c = np.stack([v.center for v in self.views])
        return int(np.argmin(np.linalg.norm(c - c.mean(0), axis=1)))


M360 = ["m360_bicycle", "m360_bonsai", "m360_counter", "m360_garden", "m360_kitchen", "m360_room", "m360_stump"]


def _root(name: str) -> tuple[Path, Path]:
    """(scene dir holding images_4 + sparse, cache file)."""
    if name.startswith("m360_"):
        d = DATA.parent / "mipnerf360" / name[5:]
        return d, d / "scene_cache.npz"
    return DATA / name / name, DATA / name / "scene_cache.npz"


def load_scene(name: str) -> Scene:
    """Load from a cached npz (no pycolmap import: it clashes with torch's libomp on macOS)."""
    root, cache = _root(name)
    if not cache.exists():
        import subprocess, sys

        subprocess.run([sys.executable, "-m", "nvs.scenes", name], check=True, cwd=DATA.parents[1])
    z = np.load(cache, allow_pickle=True)
    views = [
        View(n, root / "images_4" / p, K, w2c, pids)
        for n, p, K, w2c, pids in zip(z["names"], z["pngs"], z["Ks"], z["w2cs"], z["pids"])
    ]
    points = dict(zip(z["point_ids"].tolist(), z["point_xyz"]))
    return Scene(name, views, points, int(z["width"]), int(z["height"]))


def target_indices(scene: Scene, si: int, max_targets: int = 24, max_rot_deg: float = 35.0) -> list[int]:
    """LLFF: every other view. 360 orbits: nearest views (by centre) within max_rot_deg of the source."""
    others = [i for i in range(len(scene.views)) if i != si]
    if not scene.name.startswith("m360_"):
        return others
    sv = scene.views[si]
    c = np.stack([v.center for v in scene.views])
    keep = []
    for i in sorted(others, key=lambda i: np.linalg.norm(c[i] - c[si])):
        R = scene.views[i].w2c[:3, :3] @ sv.w2c[:3, :3].T
        if np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))) <= max_rot_deg:
            keep.append(i)
        if len(keep) == max_targets:
            break
    return sorted(keep)


def sparse_depth(scene: Scene, view: View) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP points seen by `view`, projected: returns (uv Nx2 pixel coords, z N)."""
    xyz = np.stack([scene.points[i] for i in view.point_ids])
    cam = (view.w2c[:3, :3] @ xyz.T).T + view.w2c[:3, 3]
    z = cam[:, 2]
    uv = (view.K @ (cam / z[:, None]).T).T[:, :2]
    keep = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < scene.width) & (uv[:, 1] >= 0) & (uv[:, 1] < scene.height)
    return uv[keep], z[keep]


def _build_cache(name: str) -> None:
    import pycolmap

    root, cache = _root(name)
    rec = pycolmap.Reconstruction(str(root / "sparse" / "0"))
    files = sorted(p for p in (root / "images_4").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    w, h = Image.open(files[0]).size

    images = sorted(rec.images.values(), key=lambda im: im.name)
    if name.startswith("m360_"):
        by_name = {p.name: p for p in files}  # 360 images_4 keep the original names
        pairs = [(im, by_name[im.name]) for im in images if im.name in by_name]
    else:
        if len(images) != len(files):  # LLFF images_4 are renamed image000.png in sorted order
            raise RuntimeError(f"{name}: {len(images)} COLMAP images vs {len(files)} files")
        pairs = list(zip(images, files))

    views = []
    for im, f in pairs:
        cam = rec.cameras[im.camera_id]
        K = cam.calibration_matrix().copy()
        K[0] *= w / cam.width
        K[1] *= h / cam.height
        w2c = np.eye(4)
        w2c[:3, :4] = im.cam_from_world().matrix()
        pids = np.array([p.point3D_id for p in im.points2D if p.has_point3D()], dtype=np.int64)
        views.append(View(im.name, f, K, w2c, pids))

    pids_all = np.array(list(rec.points3D.keys()), dtype=np.int64)
    np.savez(
        cache,
        names=np.array([v.name for v in views]),
        pngs=np.array([v.path.name for v in views]),
        Ks=np.stack([v.K for v in views]),
        w2cs=np.stack([v.w2c for v in views]),
        pids=np.array([v.point_ids for v in views] + [None], dtype=object)[:-1],
        point_ids=pids_all,
        point_xyz=np.stack([rec.points3D[i].xyz for i in pids_all]),
        width=w,
        height=h,
    )


if __name__ == "__main__":
    import sys

    for n in sys.argv[1:] or SCENES + M360:
        _build_cache(n)
