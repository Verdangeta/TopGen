#!/usr/bin/env python3
"""Feature independence: residual-R² and CCA between TopGen blocks and catch22/TSFresh."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import run_experiments as exp

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


OUTPUT_CSV = "results/independence/independence.csv"
TOPGEN_BLOCKS = ("b1", "b2", "b3")
PARTNERS = ("catch22", "tsfresh")
METRICS = ("residual_r2", "cca_mean_rho")
DEFAULT_N_PERM = 200
RIDGE_ALPHA = 1.0

INDEPENDENCE_FIELDS = (
    "dataset",
    "dataset_type",
    "is_dynamical",
    "block",
    "partner",
    "metric",
    "value",
    "null_mean",
    "p_value",
)


def _split_topgen_by_block(
    matrix: np.ndarray,
    records: list[dict[str, object]],
) -> dict[str, np.ndarray]:
    blocks: dict[str, list[int]] = {block: [] for block in TOPGEN_BLOCKS}
    for col_idx, record in enumerate(records):
        block = record.get("block")
        if block in blocks:
            blocks[block].append(col_idx)
    return {
        block: matrix[:, indices]
        for block, indices in blocks.items()
        if indices
    }


def _prepare_matrix(
    matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, PCA | None]:
    """Standardize; PCA-reduce when p > n."""
    n_samples, n_features = matrix.shape
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    if n_features <= max(n_samples - 2, 1):
        return scaled, None
    n_components = min(n_samples - 2, n_features)
    pca = PCA(n_components=n_components, random_state=int(rng.integers(0, 2**31 - 1)))
    return pca.fit_transform(scaled), pca


def residual_r2(
    block_matrix: np.ndarray,
    partner_matrix: np.ndarray,
    rng: np.random.Generator,
    ridge_alpha: float = RIDGE_ALPHA,
) -> float:
    """Mean 1 - R² from regressing each block column on all partner columns."""
    x_partner, _ = _prepare_matrix(partner_matrix, rng)
    x_block, _ = _prepare_matrix(block_matrix, rng)
    one_minus_r2 = []
    ridge = Ridge(alpha=ridge_alpha)
    for col_idx in range(x_block.shape[1]):
        ridge.fit(x_partner, x_block[:, col_idx])
        r2 = ridge.score(x_partner, x_block[:, col_idx])
        one_minus_r2.append(1.0 - float(np.clip(r2, 0.0, 1.0)))
    return float(np.mean(one_minus_r2))


def cca_mean_rho(
    block_matrix: np.ndarray,
    partner_matrix: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Mean of top-k canonical correlations after standardize + PCA."""
    n_samples = block_matrix.shape[0]
    x_block, _ = _prepare_matrix(block_matrix, rng)
    x_partner, _ = _prepare_matrix(partner_matrix, rng)
    k = min(5, n_samples - 2, x_block.shape[1], x_partner.shape[1])
    if k < 1:
        return 0.0
    cca = CCA(n_components=k, max_iter=1000)
    cca.fit(x_block, x_partner)
    x_block_c, x_partner_c = cca.transform(x_block, x_partner)
    rhos = []
    for comp_idx in range(k):
        rho = np.corrcoef(x_block_c[:, comp_idx], x_partner_c[:, comp_idx])[0, 1]
        if np.isfinite(rho):
            rhos.append(abs(float(rho)))
    return float(np.mean(rhos)) if rhos else 0.0


def permutation_null(
    block_matrix: np.ndarray,
    partner_matrix: np.ndarray,
    metric_fn,
    rng: np.random.Generator,
    n_perm: int,
) -> tuple[float, float, float]:
    """Observed metric, null mean, two-sided-style p-value (>= observed)."""
    observed = metric_fn(block_matrix, partner_matrix, rng)
    nulls = []
    for _ in range(n_perm):
        perm_idx = rng.permutation(partner_matrix.shape[0])
        shuffled_partner = partner_matrix[perm_idx]
        nulls.append(metric_fn(block_matrix, shuffled_partner, rng))
    nulls_arr = np.asarray(nulls, dtype=float)
    null_mean = float(np.mean(nulls_arr))
    p_value = float((1 + np.sum(nulls_arr >= observed)) / (1 + n_perm))
    return observed, null_mean, p_value


def reset_independence_csv(path: str = OUTPUT_CSV) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEPENDENCE_FIELDS)
        writer.writeheader()


def append_independence_rows(rows: list[dict[str, object]], path: str = OUTPUT_CSV) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEPENDENCE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_independence_experiments(
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
    topgen_kwargs: dict,
    n_perm: int,
    output_csv: str = OUTPUT_CSV,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    data_cache: dict[str, tuple[np.ndarray, ...]] = {}
    split_jobs = [(dataset, seed) for dataset in datasets for seed in seeds]
    last_dataset = None
    reset_independence_csv(output_csv)

    # Accumulate per (dataset, block, partner, metric) across seeds.
    accum: dict[tuple[str, str, str, str], list[tuple[float, float, float]]] = {}

    for dataset, seed in tqdm(split_jobs, desc="splits", unit="split"):
        if dataset not in data_cache:
            data_cache[dataset] = exp.load_ucr(dataset)
        train_X, train_y, _, _ = data_cache[dataset]
        rng = np.random.default_rng(seed)

        if dataset != last_dataset:
            print(f"\n{dataset}: train {train_X.shape[0]} samples (feature-feature, no labels)")
            last_dataset = dataset

        for partner in PARTNERS:
            print(f"  seed={seed}: topgen + {partner}")
            feature_blocks = exp.compute_feature_blocks(
                ("topgen", partner),
                train_X,
                train_y,
                train_X,
                seed,
                topgen_kwargs,
            )
            topgen_blocks = _split_topgen_by_block(
                feature_blocks["topgen"].train,
                feature_blocks["topgen"].records,
            )
            partner_matrix = feature_blocks[partner].train

            for block, block_matrix in topgen_blocks.items():
                for metric_name, metric_fn in (
                    ("residual_r2", residual_r2),
                    ("cca_mean_rho", cca_mean_rho),
                ):
                    observed, null_mean, p_value = permutation_null(
                        block_matrix,
                        partner_matrix,
                        metric_fn,
                        rng,
                        n_perm,
                    )
                    key = (dataset, block, partner, metric_name)
                    accum.setdefault(key, []).append((observed, null_mean, p_value))
                    print(
                        f"    {block} vs {partner} {metric_name}: "
                        f"value={observed:.4f} null_mean={null_mean:.4f} p={p_value:.4f}"
                    )

    for (dataset, block, partner, metric_name), values in sorted(accum.items()):
        observed_mean = float(np.mean([value[0] for value in values]))
        null_mean = float(np.mean([value[1] for value in values]))
        p_value = float(np.median([value[2] for value in values]))
        row = {
            "dataset": dataset,
            "dataset_type": exp.DATASET_TYPES.get(dataset, "UNKNOWN"),
            "is_dynamical": exp.is_dynamical(dataset),
            "block": block,
            "partner": partner,
            "metric": metric_name,
            "value": round(observed_mean, 6),
            "null_mean": round(null_mean, 6),
            "p_value": round(p_value, 6),
        }
        rows.append(row)
        append_independence_rows([row], output_csv)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TopGen block independence vs catch22/TSFresh (residual-R² + CCA)"
    )
    parser.add_argument("--quick", action="store_true", help="GunPoint, seed 0, n_perm=50")
    parser.add_argument(
        "--full-topgen",
        action="store_true",
        help="Use full TOPGEN_KWARGS instead of lite (default: lite).",
    )
    parser.add_argument("--output", default=None, help=f"Output CSV (default: {OUTPUT_CSV})")
    parser.add_argument("--n-perm", type=int, default=None, help="Permutation count (default 200)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, metavar="SEED")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=None, metavar="DATASET")
    args = parser.parse_args()

    use_lite = not args.full_topgen
    if args.quick:
        datasets = exp.QUICK_DATASETS
        seeds = tuple(args.seeds) if args.seeds is not None else exp.QUICK_SEEDS
        n_perm = args.n_perm if args.n_perm is not None else 50
        topgen_kwargs = exp.LITE_TOPGEN_KWARGS if use_lite else exp.TOPGEN_KWARGS
        output = args.output or "results/independence/independence_quick.csv"
    else:
        base_datasets = (
            exp.REPORT_DATASETS + exp.TUNING_DATASETS if args.all_datasets else exp.DATASETS
        )
        if args.datasets is not None:
            unknown = [name for name in args.datasets if name not in exp.DATASET_TYPES]
            if unknown:
                raise ValueError("Unknown dataset(s): " + ", ".join(unknown))
            datasets = tuple(args.datasets)
        else:
            datasets = base_datasets
        seeds = tuple(args.seeds) if args.seeds is not None else exp.SEEDS
        n_perm = args.n_perm if args.n_perm is not None else DEFAULT_N_PERM
        topgen_kwargs = exp.LITE_TOPGEN_KWARGS if use_lite else exp.TOPGEN_KWARGS
        output = args.output or OUTPUT_CSV

    print(
        f"Independence: blocks={TOPGEN_BLOCKS}, partners={PARTNERS}, "
        f"n_perm={n_perm}, topgen={'lite' if use_lite else 'full'}"
    )
    rows = run_independence_experiments(datasets, seeds, topgen_kwargs, n_perm, output)
    print(f"\nWrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
