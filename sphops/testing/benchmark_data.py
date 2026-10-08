"""Deterministic CPU particle fixtures; importing this file never initializes CUDA."""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import yaml


@dataclass
class ParticleCase:
    name: str
    family: str
    positions: np.ndarray
    boundary: str
    grid_size: tuple[int, int]
    eps: float
    seed: int
    source: dict = field(default_factory=dict)

    def metadata(self):
        return {
            "name": self.name, "family": self.family,
            "shape": list(self.positions.shape), "boundary": self.boundary,
            "grid_size": list(self.grid_size), "eps": self.eps,
            "seed": self.seed, "source": self.source,
            "positions_sha256": hashlib.sha256(self.positions.tobytes()).hexdigest(),
        }


def disk_positions(rng, batch_size, num_particles, radius):
    theta = rng.uniform(0.0, 2.0 * np.pi, (batch_size, num_particles))
    radius_sample = np.sqrt(rng.random((batch_size, num_particles))) * radius
    return np.ascontiguousarray(
        np.stack((np.cos(theta) * radius_sample, np.sin(theta) * radius_sample), -1),
        dtype=np.float32,
    )


def _read_image(source, root):
    if "path" in source:
        path = root / source["path"]
        with Image.open(path) as image:
            return image.convert("RGBA")
    archive_path = root / source["archive"]
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(source["member"]) as data, Image.open(data) as image:
            return image.convert("RGBA")


def _fit_image(image, size):
    # Keep aspect ratio, including when cached references need upsampling.
    scale = size / max(image.size)
    shape = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(shape, Image.Resampling.LANCZOS)


def image_positions(image, rng, batch_size, num_particles, threshold=0.05):
    """Sample unique alpha-mask pixels in [-1, 1]^2 at an adaptive resolution.

    As in training, resolution follows the number of opaque pixels. The final
    resolution is increased if necessary so fixed-N fixtures do not introduce
    coincident particles by sampling with replacement.
    """
    if not 0.0 <= threshold < 1.0:
        raise ValueError("alpha threshold must be in [0, 1)")
    size = 128
    for _ in range(5):
        resized = _fit_image(image, size)
        alpha = np.asarray(resized)[..., 3] / 255.0
        count = int(np.count_nonzero(alpha > threshold))
        if count == 0:
            raise ValueError("Target image has no pixels above the alpha threshold")
        size = max(8, round(size * np.sqrt(num_particles / count)))
    for _ in range(16):
        if size > 4096:
            raise ValueError("Target requires a resolution above 4096 pixels")
        resized = _fit_image(image, size)
        alpha = np.asarray(resized)[..., 3] / 255.0
        y, x = np.nonzero(alpha > threshold)
        if len(x) >= num_particles:
            break
        if len(x) == 0:
            raise ValueError("Target image has no pixels above the alpha threshold")
        size = max(size + 1, int(np.ceil(size * np.sqrt(num_particles / len(x)))))
    else:
        raise ValueError("Could not obtain enough distinct target pixels")
    candidates = np.stack(
        (-1.0 + 2.0 * (x + 0.5) / resized.width,
         1.0 - 2.0 * (y + 0.5) / resized.height), -1,
    ).astype(np.float32)
    selected = np.stack([
        rng.choice(len(candidates), num_particles, replace=False)
        for _ in range(batch_size)
    ])
    return np.ascontiguousarray(candidates[selected]), {
        "image_size": list(resized.size), "eligible_pixels": len(candidates),
        "alpha_threshold": threshold, "replacement": False,
    }


def build_cases(manifest, *, batch_size=8, num_particles=4096, seed=42,
                selected=None, include_intermediate=True):
    if batch_size < 1 or num_particles < 1:
        raise ValueError("batch_size and num_particles must be positive")
    manifest = Path(manifest).resolve()
    with manifest.open() as data:
        config = yaml.safe_load(data)
    root = manifest.parent / config.get("asset_root", "../..")
    cases = []
    specs = [
        ("growing_seed", "growing_seed", "clamped", (16, 16), 0.1, 0.2),
        ("texture_seed", "texture_seed", "periodic", (10, 10), 0.2, None),
    ]
    if include_intermediate:
        specs += [
            ("growing_disk_04", "growing_transition", "clamped", (16, 16), 0.1, 0.4),
            ("growing_disk_06", "growing_transition", "clamped", (16, 16), 0.1, 0.6),
        ]
    for index, (name, family, boundary, grid_size, eps, radius) in enumerate(specs):
        if selected and name not in selected:
            continue
        case_seed = seed + index
        rng = np.random.default_rng(case_seed)
        if radius is None:
            positions = rng.uniform(-1, 1, (batch_size, num_particles, 2)).astype(np.float32)
        else:
            positions = disk_positions(rng, batch_size, num_particles, radius)
        cases.append(ParticleCase(
            name, family, positions, boundary, grid_size, eps, case_seed,
            {"radius": radius} if radius is not None else {"aabb": [-1, 1, -1, 1]},
        ))
    for family, entries, boundary, grid_size, eps, threshold in [
        ("emoji_target", config.get("emoji_targets", []), "clamped", (16, 16), 0.1, 0.05),
        ("texture_target", config.get("texture_targets", []), "periodic", (10, 10), 0.2, 0.02),
    ]:
        for index, entry in enumerate(entries):
            name = entry["name"]
            if selected and name not in selected:
                continue
            case_seed = seed + (100 if family == "emoji_target" else 200) + index
            source = {key: value for key, value in entry.items() if key != "name"}
            try:
                image = _read_image(source, root)
            except (FileNotFoundError, KeyError) as error:
                raise ValueError(f"Missing asset for {name}: {error}; edit {manifest}") from error
            positions, sampling = image_positions(
                image, np.random.default_rng(case_seed), batch_size, num_particles, threshold,
            )
            cases.append(ParticleCase(
                name, family, positions, boundary, grid_size, eps, case_seed,
                {**source, **sampling},
            ))
    if selected:
        missing = set(selected) - {case.name for case in cases}
        if missing:
            raise ValueError(f"Unknown cases: {', '.join(sorted(missing))}")
    return cases


def neighbor_statistics(case, queries=256):
    """Geometry statistics only; not CUDA timing results."""
    positions = case.positions[0]
    rng = np.random.default_rng(case.seed + 1000)
    query = positions[rng.choice(len(positions), min(queries, len(positions)), replace=False)]
    delta = positions[None] - query[:, None]
    if case.boundary == "periodic":
        length = np.asarray(case.grid_size) * case.eps
        delta = (delta + length / 2) % length - length / 2
    neighbors = np.sum(np.sum(delta * delta, -1) < case.eps ** 2, -1)
    cells = np.floor(positions / case.eps).astype(np.int32) + np.asarray(case.grid_size) // 2
    if case.boundary == "periodic":
        cells %= np.asarray(case.grid_size)
    else:
        cells = np.clip(cells, 0, np.asarray(case.grid_size) - 1)
    occupancy = np.bincount(
        cells[:, 1] * case.grid_size[0] + cells[:, 0], minlength=int(np.prod(case.grid_size)),
    )
    return {
        "sampled_queries": len(query), "neighbors_mean": float(neighbors.mean()),
        "neighbors_p95": float(np.percentile(neighbors, 95)),
        "occupied_cells": int(np.count_nonzero(occupancy)),
        "max_cell_occupancy": int(occupancy.max()),
    }
