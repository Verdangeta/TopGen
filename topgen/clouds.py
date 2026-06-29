"""Delay embedding, cloud construction, and provenance-tracked subsampling."""

from __future__ import annotations

import numpy as np
from gtda.time_series import TakensEmbedding
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def embed_series(series: np.ndarray, embedder: TakensEmbedding) -> np.ndarray:
    """Delay-embed one univariate series via a fitted giotto-tda embedder."""
    series = np.asarray(series, dtype=float).ravel()
    return embedder.fit_transform(series.reshape(1, -1))[0]


def subsample(points: np.ndarray, s: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Subsample up to s points using maxmin coverage or uniform random draws."""
    points = np.asarray(points, dtype=float)
    if points.shape[0] <= s:
        return points.copy()
    if mode == "uniform":
        idx = rng.choice(points.shape[0], size=s, replace=False)
        return points[idx]
    if mode == "maxmin":
        return _maxmin_subsample(points, s, rng)
    raise ValueError(f"Unknown subsample mode: {mode}")


def _maxmin_subsample(points: np.ndarray, s: int, rng: np.random.Generator) -> np.ndarray:
    """Farthest-point (maxmin) subsampling for coverage."""
    n_points = points.shape[0]
    selected = [int(rng.integers(n_points))]
    dists = pairwise_distances(points, points[selected]).ravel()
    for _ in range(s - 1):
        next_idx = int(np.argmax(dists))
        selected.append(next_idx)
        new_dists = pairwise_distances(points, points[[next_idx]]).ravel()
        dists = np.minimum(dists, new_dists)
    return points[selected]


def build_series_cloud(
    series: np.ndarray,
    embedder: TakensEmbedding,
    s: int,
    subsample_mode: str,
    pca: PCA,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Embed one series, PCA-project, and subsample to budget s.

    When ``seed`` is given the subsample uses a fresh deterministic generator so
    the same series always yields the same cloud (no dependence on call order).
    """
    embedded = embed_series(series, embedder)
    projected = pca.transform(embedded)
    if seed is not None:
        rng = np.random.default_rng(seed)
    return subsample(projected, s, subsample_mode, rng)


def build_class_cloud(
    series_list: list[np.ndarray],
    source_indices: list[int],
    embedder: TakensEmbedding,
    s: int,
    subsample_mode: str,
    pca: PCA,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool per-series clouds into one class cloud with provenance indices."""
    chunks = []
    provenance = []
    for series, src_idx in zip(series_list, source_indices):
        cloud = build_series_cloud(series, embedder, s, subsample_mode, pca, rng)
        chunks.append(cloud)
        provenance.append(np.full(cloud.shape[0], src_idx, dtype=int))
    if not chunks:
        return np.empty((0, pca.n_components_)), np.empty(0, dtype=int)
    return np.vstack(chunks), np.concatenate(provenance)


def sample_subcloud(
    cloud: np.ndarray,
    provenance: np.ndarray,
    size: int,
    mode: str,
    rng: np.random.Generator | None = None,
    exclude_series: int | None = None,
    class_mode: str = "A",
    seed: int | None = None,
) -> np.ndarray:
    """Draw class-side subsample C_c' with optional provenance exclusion.

    With ``seed`` the draw is deterministic for a given (class, exclusion): the
    same class always yields the same right cloud, so identical query series get
    identical features. If the available points are at or below ``size`` they are
    all returned (no subsampling), which keeps the right cloud size-matched.
    """
    if cloud.shape[0] == 0:
        return cloud.copy()
    if seed is not None:
        rng = np.random.default_rng(seed)
    mask = np.ones(cloud.shape[0], dtype=bool)
    if exclude_series is not None:
        mask &= provenance != exclude_series
    available = cloud[mask]
    if available.shape[0] == 0:
        return available
    if class_mode == "B":
        return _mode_b_subsample(cloud, provenance, mask, size, mode, rng)
    return subsample(available, size, mode, rng)


def _mode_b_subsample(
    cloud: np.ndarray,
    provenance: np.ndarray,
    mask: np.ndarray,
    size: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """At most one point per source series (Mode B)."""
    available_idx = np.where(mask)[0]
    by_series: dict[int, list[int]] = {}
    for idx in available_idx:
        by_series.setdefault(int(provenance[idx]), []).append(idx)
    series_ids = list(by_series.keys())
    rng.shuffle(series_ids)
    chosen_idx = []
    for series_id in series_ids[:size]:
        point_idx = by_series[series_id]
        if mode == "uniform":
            chosen_idx.append(int(rng.choice(point_idx)))
        else:
            local_points = cloud[point_idx]
            chosen_idx.append(point_idx[int(np.argmax(pairwise_distances(local_points).sum(axis=1)))])
    return cloud[chosen_idx]
