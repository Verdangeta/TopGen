"""Cross-persistence linear representations, feature blocks, and self-densities."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass

import mtd
import numpy as np
from scipy.stats import entropy, gaussian_kde


# All seven linear representations from methodology §6 (per homology dimension).
ALL_REP_NAMES: tuple[str, ...] = (
    "mtd",
    "total_persistence",
    "pers_entropy",
    "landscape_l2",
    "betti_0",
    "betti_1",
    "betti_2",
)
LITE_REP_NAMES: tuple[str, ...] = tuple(rep for rep in ALL_REP_NAMES if rep != "betti_2")

REPRESENTATIONS = ALL_REP_NAMES

# Betti thresholds are quantiles of the lifetimes observed at fit time (frozen),
# not absolute cutoffs: PCA-space lifetimes are unnormalised so a fixed cutoff
# produces dead (all-constant) features. See betti_thresholds_from_lifetimes.
BETTI_QUANTILES = (0.25, 0.5, 0.75)


@dataclass
class FrozenDensity:
    """Self-density objects fit once per (class, representation, homology dim)."""

    kde: gaussian_kde | None
    sorted_values: np.ndarray


def _digest_cloud(hasher: "hashlib._Hash", cloud: np.ndarray) -> None:
    """Fold one point cloud into a content hash (bytes + shape + dtype)."""
    arr = np.ascontiguousarray(np.asarray(cloud, dtype=float))
    hasher.update(repr(arr.shape).encode())
    hasher.update(repr(arr.dtype).encode())
    hasher.update(arr.tobytes())


def barcode_cache_key(
    left: np.ndarray,
    right: np.ndarray,
    batch_size_left: int,
    batch_size_right: int,
    pdist_device: str,
) -> str:
    """Content-addressed key from both clouds and MTopDiv batch/device args."""
    hasher = hashlib.sha256()
    _digest_cloud(hasher, left)
    _digest_cloud(hasher, right)
    hasher.update(repr(int(batch_size_left)).encode())
    hasher.update(repr(int(batch_size_right)).encode())
    hasher.update(pdist_device.encode())
    return hasher.hexdigest()


def _barcode_cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"{key}.npz")


def _barcode_to_npz_arrays(barcode) -> dict[str, np.ndarray]:
    """Serialize H0/H1 birth–death arrays for disk storage."""
    arrays: dict[str, np.ndarray] = {}
    for hom_dim, bars in enumerate(barcode):
        bars = np.asarray(bars, dtype=float)
        if bars.size == 0:
            arrays[f"h{hom_dim}"] = np.empty((0, 2), dtype=float)
        else:
            arrays[f"h{hom_dim}"] = bars.reshape(-1, 2)
    arrays["n_hom"] = np.array([len(barcode)], dtype=np.int32)
    return arrays


def _barcode_from_npz(payload) -> np.ndarray:
    """Reconstruct MTopDiv barcode as object-array to preserve .shape API."""
    n_hom = int(np.asarray(payload["n_hom"]).ravel()[0])
    bars = [np.asarray(payload[f"h{hom_dim}"], dtype=float) for hom_dim in range(n_hom)]
    return np.asarray(bars, dtype=object)


def _load_barcode_cache(cache_path: str):
    if not os.path.isfile(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            return _barcode_from_npz(payload)
    except (OSError, ValueError, KeyError, EOFError):
        return None


def _save_barcode_cache(cache_path: str, barcode) -> None:
    """Atomic write: temp file in cache dir, then os.replace."""
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_base = tempfile.mkstemp(dir=cache_dir)
    os.close(fd)
    os.unlink(tmp_base)
    tmp_path = f"{tmp_base}.npz"
    try:
        np.savez_compressed(tmp_base, **_barcode_to_npz_arrays(barcode))
        os.replace(tmp_path, cache_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _compute_cross_barcode(
    left: np.ndarray,
    right: np.ndarray,
    batch_size_left: int,
    batch_size_right: int,
    pdist_device: str,
):
    """Raw MTopDiv cross-barcode (same convention as Topological_classifier)."""
    return mtd.calc_cross_barcodes(
        left,
        right,
        batch_size1=batch_size_left,
        batch_size2=batch_size_right,
        pdist_device=pdist_device,
        is_plot=False,
    )


def cross_barcode(
    left: np.ndarray,
    right: np.ndarray,
    batch_size_left: int,
    batch_size_right: int,
    pdist_device: str = "cuda",
    cache_dir: str | None = None,
) -> np.ndarray:
    """Wrap MTopDiv cross-barcode computation with optional content-addressed cache.

    When ``cache_dir`` is set, barcodes are keyed by the raw left/right point
    clouds (plus batch sizes and device). Different ``random_state`` values that
    produce different clouds automatically get different keys.
    """
    if cache_dir is None:
        return _compute_cross_barcode(
            left, right, batch_size_left, batch_size_right, pdist_device
        )

    key = barcode_cache_key(left, right, batch_size_left, batch_size_right, pdist_device)
    cache_path = _barcode_cache_path(cache_dir, key)
    cached = _load_barcode_cache(cache_path)
    if cached is not None:
        return cached

    barcode = _compute_cross_barcode(
        left, right, batch_size_left, batch_size_right, pdist_device
    )
    _save_barcode_cache(cache_path, barcode)
    return barcode


def _homology_bars(barcode: np.ndarray, hom_dim: int) -> np.ndarray:
    """Return (n, 2) birth-death array for homology dimension hom_dim."""
    if len(barcode) <= hom_dim:
        return np.empty((0, 2), dtype=float)
    bars = np.asarray(barcode[hom_dim], dtype=float)
    if bars.size == 0:
        return np.empty((0, 2), dtype=float)
    return bars.reshape(-1, 2)


def _lifetimes(bars: np.ndarray, hom_dim: int) -> np.ndarray:
    """Barcode lifetimes; H0 bars are death-only (births are zero)."""
    if bars.shape[0] == 0:
        return np.empty(0, dtype=float)
    lifetimes = bars[:, 1] - bars[:, 0]
    finite = lifetimes[np.isfinite(lifetimes)]
    if finite.size:
        lifetimes = np.where(np.isfinite(lifetimes), lifetimes, np.max(finite))
    else:
        lifetimes = np.zeros_like(lifetimes)
    return np.clip(lifetimes, 0.0, 1e6)


def barcode_lifetimes(barcode: np.ndarray, hom_dim: int) -> np.ndarray:
    """Lifetimes of a cross-barcode at one homology dim (H0 bars are death-only)."""
    return _lifetimes(_homology_bars(barcode, hom_dim), hom_dim)


def betti_thresholds_from_lifetimes(
    lifetimes_by_hom: dict[int, np.ndarray],
    quantiles: tuple[float, ...] = BETTI_QUANTILES,
) -> dict[int, np.ndarray]:
    """Freeze per-homology Betti thresholds as quantiles of observed lifetimes."""
    thresholds: dict[int, np.ndarray] = {}
    for hom_dim, lifetimes in lifetimes_by_hom.items():
        lifetimes = np.asarray(lifetimes, dtype=float)
        lifetimes = lifetimes[np.isfinite(lifetimes)]
        if lifetimes.size == 0:
            thresholds[hom_dim] = np.zeros(len(quantiles), dtype=float)
        else:
            thresholds[hom_dim] = np.quantile(lifetimes, quantiles)
    return thresholds


def linear_reps(
    barcode: np.ndarray,
    hom_dims: tuple[int, ...] = (0, 1),
    betti_thresholds: dict[int, np.ndarray] | None = None,
) -> dict[tuple[str, int], float]:
    """Scalar linear representations for each homology dimension.

    ``betti_thresholds`` maps a homology dim to its frozen quantile cutoffs. When
    omitted, thresholds fall back to this barcode's own lifetime quantiles.
    """
    reps: dict[tuple[str, int], float] = {}
    for hom_dim in hom_dims:
        bars = _homology_bars(barcode, hom_dim)
        lifetimes = _lifetimes(bars, hom_dim)
        reps[("mtd", hom_dim)] = _clip_score(mtd.get_score(barcode, hom_dim, "sum_length"))
        reps[("total_persistence", hom_dim)] = _clip_score(
            mtd.get_score(barcode, hom_dim, "sum_sq_length")
        )
        reps[("pers_entropy", hom_dim)] = float(entropy(lifetimes, base=2)) if lifetimes.size else 0.0
        reps[("landscape_l2", hom_dim)] = _landscape_l2_norm(lifetimes)
        if betti_thresholds is not None and hom_dim in betti_thresholds:
            cutoffs = betti_thresholds[hom_dim]
        elif lifetimes.size:
            cutoffs = np.quantile(lifetimes, BETTI_QUANTILES)
        else:
            cutoffs = np.zeros(len(BETTI_QUANTILES), dtype=float)
        for idx, threshold in enumerate(cutoffs):
            reps[(f"betti_{idx}", hom_dim)] = (
                float(np.sum(lifetimes >= threshold)) if lifetimes.size else 0.0
            )
    return reps


def _clip_score(value: float) -> float:
    if not np.isfinite(value):
        return 1e6
    return float(np.clip(value, -1e6, 1e6))


def _landscape_l2_norm(lifetimes: np.ndarray) -> float:
    """‖λ₁‖₂ surrogate: L2 norm of sorted lifetimes (landscape peak proxy)."""
    if lifetimes.size == 0:
        return 0.0
    return float(np.linalg.norm(np.sort(lifetimes), ord=2))


def scalar_rep_value(
    barcode: np.ndarray,
    rep_name: str,
    hom_dim: int,
    betti_thresholds: dict[int, np.ndarray] | None = None,
) -> float:
    """Evaluate one representation on one cross-barcode."""
    return linear_reps(barcode, hom_dims=(hom_dim,), betti_thresholds=betti_thresholds)[
        (rep_name, hom_dim)
    ]


def self_density_fit(values: np.ndarray) -> FrozenDensity:
    """Boundary-corrected 1-D KDE (reflect at 0) plus empirical CDF storage."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return FrozenDensity(kde=None, sorted_values=np.array([0.0]))
    sorted_values = np.sort(values)
    kde = None
    if np.unique(sorted_values).size > 1:
        reflected = np.concatenate([-sorted_values, sorted_values])
        try:
            kde = gaussian_kde(reflected)
        except np.linalg.LinAlgError:
            kde = None
    return FrozenDensity(kde=kde, sorted_values=sorted_values)


def self_density_tail(
    density: FrozenDensity,
    value: float,
    estimator: str = "kde",
) -> float:
    """One-sided upper-tail probability P(X >= value) under the frozen self-density."""
    if not np.isfinite(value):
        return 1.0
    if estimator == "ecdf":
        sorted_values = density.sorted_values
        if sorted_values.size == 0:
            return 1.0
        count_ge = np.searchsorted(sorted_values, value, side="left")
        return float(1.0 - count_ge / sorted_values.size)
    if estimator == "kde":
        if density.kde is None:
            return self_density_tail(density, value, estimator="ecdf")
        grid = np.linspace(0.0, max(value, density.sorted_values.max()) * 1.2 + 1e-6, 512)
        pdf = density.kde(grid)
        pdf = np.maximum(pdf, 0.0)
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1] if cdf[-1] > 0 else 1.0
        idx = np.searchsorted(grid, value, side="left")
        tail = 1.0 - (cdf[idx - 1] if idx > 0 else 0.0)
        return float(np.clip(tail, 0.0, 1.0))
    raise ValueError(f"Unknown density estimator '{estimator}'")


def assemble_blocks(
    barcode_qc: np.ndarray,
    barcode_cq: np.ndarray,
    densities: dict[tuple[str, int], FrozenDensity],
    rep_names: tuple[str, ...],
    hom_dims: tuple[int, ...],
    blocks: tuple[str, ...],
    betti_thresholds: dict[int, np.ndarray] | None = None,
    density_estimator: str = "kde",
) -> np.ndarray:
    """Per-class feature slice from precomputed barcodes: B1 raw, B2 asym, B3 membership."""
    features: list[float] = []
    for hom_dim in hom_dims:
        for rep_name in rep_names:
            val_qc = scalar_rep_value(barcode_qc, rep_name, hom_dim, betti_thresholds)
            val_cq = scalar_rep_value(barcode_cq, rep_name, hom_dim, betti_thresholds)
            if "b1" in blocks:
                features.extend([val_qc, val_cq])
            if "b2" in blocks:
                features.append(val_qc - val_cq)
            if "b3" in blocks:
                density = densities.get((rep_name, hom_dim))
                if density is None:
                    features.append(1.0)
                else:
                    features.append(self_density_tail(density, val_qc, estimator=density_estimator))
    return np.asarray(features, dtype=float)


def feature_blocks(
    query_cloud: np.ndarray,
    class_subcloud: np.ndarray,
    densities: dict[tuple[str, int], FrozenDensity],
    rep_names: tuple[str, ...],
    hom_dims: tuple[int, ...],
    blocks: tuple[str, ...],
    query_size: int,
    class_size: int,
    pdist_device: str = "cuda",
    density_estimator: str = "kde",
    betti_thresholds: dict[int, np.ndarray] | None = None,
    cache_dir: str | None = None,
) -> np.ndarray:
    """Compute both-order cross-barcodes for a query/class pair, then assemble blocks."""
    barcode_qc = cross_barcode(
        query_cloud, class_subcloud, query_size, class_size, pdist_device, cache_dir=cache_dir
    )
    barcode_cq = cross_barcode(
        class_subcloud, query_cloud, class_size, query_size, pdist_device, cache_dir=cache_dir
    )
    return assemble_blocks(
        barcode_qc,
        barcode_cq,
        densities,
        rep_names,
        hom_dims,
        blocks,
        betti_thresholds=betti_thresholds,
        density_estimator=density_estimator,
    )
