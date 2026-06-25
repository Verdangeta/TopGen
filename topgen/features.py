"""Cross-persistence linear representations, feature blocks, and self-densities."""

from __future__ import annotations

from dataclasses import dataclass

import mtd
import numpy as np
from scipy.stats import entropy, gaussian_kde

from topgen.clouds import sample_disjoint_pair


REPRESENTATIONS = ("mtd", "total_persistence", "pers_entropy", "landscape_l2", "betti")

# Fixed Betti thresholds (methodology §6); tuned once, not per dataset.
BETTI_THRESHOLDS = (0.25, 0.5, 0.75)


@dataclass
class FrozenDensity:
    """Self-density objects fit once per (class, representation, homology dim)."""

    kde: gaussian_kde | None
    sorted_values: np.ndarray


def cross_barcode(
    left: np.ndarray,
    right: np.ndarray,
    batch_size_left: int,
    batch_size_right: int,
    pdist_device: str = "cuda",
) -> np.ndarray:
    """Wrap MTopDiv cross-barcode computation (same convention as Topological_classifier)."""
    return mtd.calc_cross_barcodes(
        left,
        right,
        batch_size1=batch_size_left,
        batch_size2=batch_size_right,
        pdist_device=pdist_device,
        is_plot=False,
    )


def _homology_bars(barcode: np.ndarray, hom_dim: int) -> np.ndarray:
    """Return (n, 2) birth-death array for homology dimension hom_dim."""
    if barcode.shape[0] <= hom_dim:
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


def linear_reps(barcode: np.ndarray, hom_dims: tuple[int, ...] = (0, 1)) -> dict[tuple[str, int], float]:
    """Scalar linear representations for each homology dimension."""
    reps: dict[tuple[str, int], float] = {}
    for hom_dim in hom_dims:
        bars = _homology_bars(barcode, hom_dim)
        lifetimes = _lifetimes(bars, hom_dim)
        reps[("mtd", hom_dim)] = _clip_score(mtd.get_score(barcode, hom_dim, "sum_length"))
        reps[("total_persistence", hom_dim)] = float(np.sum(lifetimes ** 2)) if lifetimes.size else 0.0
        reps[("pers_entropy", hom_dim)] = float(entropy(lifetimes, base=2)) if lifetimes.size else 0.0
        reps[("landscape_l2", hom_dim)] = _landscape_l2_norm(lifetimes)
        for idx, threshold in enumerate(BETTI_THRESHOLDS):
            reps[(f"betti_{idx}", hom_dim)] = float(np.sum(lifetimes >= threshold)) if lifetimes.size else 0.0
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
) -> float:
    """Evaluate one representation on one cross-barcode."""
    return linear_reps(barcode, hom_dims=(hom_dim,))[(rep_name, hom_dim)]


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
) -> np.ndarray:
    """Assemble per-class feature slice: B1 raw, B2 asymmetry, B3 membership."""
    features: list[float] = []
    barcode_qc = cross_barcode(query_cloud, class_subcloud, query_size, class_size, pdist_device)
    barcode_cq = cross_barcode(class_subcloud, query_cloud, class_size, query_size, pdist_device)

    for hom_dim in hom_dims:
        for rep_name in rep_names:
            val_qc = scalar_rep_value(barcode_qc, rep_name, hom_dim)
            val_cq = scalar_rep_value(barcode_cq, rep_name, hom_dim)
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


def estimate_class_self_densities(
    class_cloud: np.ndarray,
    class_provenance: np.ndarray,
    rep_names: tuple[str, ...],
    hom_dims: tuple[int, ...],
    left_size: int,
    right_size: int,
    n_samples: int,
    subsample_mode: str,
    rng: np.random.Generator,
    pdist_device: str = "cuda",
) -> dict[tuple[str, int], FrozenDensity]:
    """Fit frozen self-densities from size-matched disjoint intra-class subsample pairs."""
    collected: dict[tuple[str, int], list[float]] = {
        (rep_name, hom_dim): [] for rep_name in rep_names for hom_dim in hom_dims
    }
    for _ in range(n_samples):
        left, right = sample_disjoint_pair(
            class_cloud,
            class_provenance,
            left_size,
            right_size,
            subsample_mode,
            rng,
        )
        if left.shape[0] == 0 or right.shape[0] == 0:
            continue
        barcode = cross_barcode(left, right, left.shape[0], right.shape[0], pdist_device)
        for hom_dim in hom_dims:
            for rep_name in rep_names:
                collected[(rep_name, hom_dim)].append(
                    scalar_rep_value(barcode, rep_name, hom_dim)
                )
    densities: dict[tuple[str, int], FrozenDensity] = {}
    for key, values in collected.items():
        densities[key] = self_density_fit(np.asarray(values, dtype=float))
    return densities
