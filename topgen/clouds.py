"""Delay embedding, cloud construction, and provenance-tracked subsampling."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def _mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 64) -> float:
    """Histogram-based mutual information between two 1-D signals."""
    hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nonzero = pxy > 0
    return float(np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero])))


def estimate_tau(series: np.ndarray, tau_max: int = 50) -> int:
    """First minimum of auto-mutual information (Takens time delay)."""
    series = np.asarray(series, dtype=float).ravel()
    mis = []
    for tau in range(1, min(tau_max, len(series) // 10) + 1):
        mis.append(_mutual_information(series[:-tau], series[tau:]))
    if len(mis) < 3:
        return 1
    # First local minimum after the initial decline.
    for idx in range(1, len(mis) - 1):
        if mis[idx] < mis[idx - 1] and mis[idx] < mis[idx + 1]:
            return idx + 1
    return int(np.argmin(mis)) + 1


def estimate_embedding_dimension(
    series: np.ndarray,
    tau: int,
    dim_max: int = 10,
    rtol: float = 10.0,
    atol: float = 2.0,
) -> int:
    """False-nearest-neighbors estimate of Takens embedding dimension."""
    series = np.asarray(series, dtype=float).ravel()
    for dimension in range(1, dim_max + 1):
        embedded = delay_embed(series, tau, dimension)
        if embedded.shape[0] < 2:
            continue
        dists = pairwise_distances(embedded)
        np.fill_diagonal(dists, np.inf)
        nn_idx = np.argmin(dists, axis=1)
        nn_dists = dists[np.arange(len(nn_idx)), nn_idx]

        embedded_next = delay_embed(series, tau, dimension + 1)
        if embedded_next.shape[0] != embedded.shape[0]:
            break
        diff = embedded_next - embedded_next[nn_idx]
        sep = np.linalg.norm(diff, axis=1)
        ratio = sep / np.maximum(nn_dists, 1e-12)
        false_rate = np.mean((ratio > rtol) | (sep / np.std(series) > atol))
        if false_rate < 0.02:
            return dimension
    return min(dim_max, 3)


def delay_embed(series: np.ndarray, tau: int, m: int) -> np.ndarray:
    """Takens delay embedding without striding."""
    series = np.asarray(series, dtype=float).ravel()
    n_points = len(series) - (m - 1) * tau
    if n_points <= 0:
        raise ValueError("Series too short for the requested embedding parameters")
    out = np.empty((n_points, m), dtype=float)
    for dim in range(m):
        start = (m - 1 - dim) * tau
        out[:, dim] = series[start : start + n_points]
    return out


def embed(series: np.ndarray, tau: int, m: int, stride: int = 1) -> np.ndarray:
    """Delay embedding followed by striding."""
    points = delay_embed(series, tau, m)
    return points[::stride]


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
    tau: int,
    m_embed: int,
    stride: int,
    s: int,
    subsample_mode: str,
    pca: PCA,
    rng: np.random.Generator,
) -> np.ndarray:
    """Embed one series, PCA-project, and subsample to budget s."""
    embedded = embed(series, tau, m_embed, stride=stride)
    projected = pca.transform(embedded)
    return subsample(projected, s, subsample_mode, rng)


def build_class_cloud(
    series_list: list[np.ndarray],
    source_indices: list[int],
    tau: int,
    m_embed: int,
    stride: int,
    s: int,
    subsample_mode: str,
    pca: PCA,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool per-series clouds into one class cloud with provenance indices."""
    chunks = []
    provenance = []
    for series, src_idx in zip(series_list, source_indices):
        cloud = build_series_cloud(series, tau, m_embed, stride, s, subsample_mode, pca, rng)
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
    rng: np.random.Generator,
    exclude_series: int | None = None,
    class_mode: str = "A",
) -> np.ndarray:
    """Draw class-side subsample C_c' with optional provenance exclusion."""
    if cloud.shape[0] == 0:
        return cloud.copy()
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


def sample_disjoint_pair(
    cloud: np.ndarray,
    provenance: np.ndarray,
    left_size: int,
    right_size: int,
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint subsamples for self-density estimation."""
    if cloud.shape[0] < left_size + right_size:
        left = subsample(cloud, min(left_size, cloud.shape[0]), mode, rng)
        remaining_mask = np.ones(cloud.shape[0], dtype=bool)
        # Greedy exclusion: drop points closest to the left subsample indices.
        left_idx = _nearest_indices(cloud, left)
        remaining_mask[left_idx] = False
        right_pool = cloud[remaining_mask]
        right = subsample(right_pool, min(right_size, right_pool.shape[0]), mode, rng)
        return left, right
    perm = rng.permutation(cloud.shape[0])
    left_idx = perm[:left_size]
    right_idx = perm[left_size : left_size + right_size]
    return cloud[left_idx], cloud[right_idx]


def _nearest_indices(cloud: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """Indices of cloud points that coincide with rows of subset (approximate)."""
    if subset.shape[0] == 0:
        return np.array([], dtype=int)
    dists = pairwise_distances(cloud, subset)
    return np.unique(np.argmin(dists, axis=0))
