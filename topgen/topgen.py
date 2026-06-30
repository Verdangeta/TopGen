"""Scikit-learn compatible TopGen v2 population-level feature generator."""

from __future__ import annotations

import time
import warnings
import zlib

import numpy as np
from gtda.time_series import SingleTakensEmbedding, TakensEmbedding
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted

from topgen.clouds import (
    build_class_cloud,
    build_series_cloud,
    cloud_budget,
    embed_series,
    sample_subcloud,
)
from topgen.features import (
    ALL_REP_NAMES,
    assemble_blocks,
    barcode_lifetimes,
    betti_thresholds_from_lifetimes,
    cross_barcode,
    feature_blocks,
    scalar_rep_value,
    self_density_fit,
)


def _embedding_search_caps(
    series_length: int,
    max_time_delay: int,
    max_dimension: int,
    stride: int,
) -> tuple[int, int]:
    """Cap giotto-tda search bounds so FNN trials fit the series length.

    gtda needs ``n_timestamps > time_delay * (dimension - 1) + 1`` and evaluates
    false-nearest-neighbours up to ``max_dimension + 2``. FNN also requires at
    least two embedded points (``kneighbors`` with ``n_neighbors=2``).
    """
    td_cap = max(1, max_time_delay)
    dim_cap = max(2, max_dimension)
    while td_cap >= 1:
        # Largest d with time_delay * (d - 1) <= series_length - 1 - stride.
        max_fit_dim = (series_length - 1 - stride) // max(td_cap, 1) + 1
        if dim_cap + 2 <= max_fit_dim:
            break
        dim_cap -= 1
        if dim_cap < 2:
            dim_cap = 2
            td_cap -= 1
    return max(1, td_cap), max(2, dim_cap)


def _embedded_point_count(series_length: int, time_delay: int, dimension: int, stride: int) -> int:
    """Number of Takens vectors for a univariate series of given length."""
    span = time_delay * (dimension - 1)
    if series_length <= span:
        return 0
    return (series_length - span) // stride


def _cap_fixed_embedding(
    series_length: int,
    time_delay: int,
    dimension: int,
    stride: int,
    min_embedded_points: int = 5,
) -> tuple[int, int, int]:
    """Reduce (tau, m, stride) so Takens embedding fits and yields enough points."""
    td = max(1, int(time_delay))
    dim = max(2, int(dimension))
    s = max(1, int(stride))

    def fits(t: int, d: int, st: int) -> bool:
        return series_length > t * (d - 1) + 1

    def viable(t: int, d: int, st: int) -> bool:
        return fits(t, d, st) and _embedded_point_count(series_length, t, d, st) >= min_embedded_points

    if viable(td, dim, s):
        return td, dim, s

    for s_try in range(s, 0, -1):
        for d_try in range(dim, 1, -1):
            for t_try in range(td, 0, -1):
                if viable(t_try, d_try, s_try):
                    return t_try, d_try, s_try

    for s_try in range(s, 0, -1):
        for d_try in range(dim, 1, -1):
            for t_try in range(td, 0, -1):
                if fits(t_try, d_try, s_try):
                    return t_try, d_try, s_try

    raise ValueError(
        f"Series length {series_length} is too short for any Takens embedding "
        f"(requested time_delay={time_delay}, dimension={dimension}, stride={stride})."
    )


class TopGenTransformer(BaseEstimator, TransformerMixin):
    """Population-level topological features via cross-persistence between class clouds."""

    def __init__(
        self,
        vectorizer: str = "embedding",
        embedding_dimension: int = 50,
        embedding_time_delay: int = 4,
        search_embedding: bool = True,
        stride: int = 5,
        n_components: int = 3,
        per_series_fraction: float = 0.8,
        query_fraction: float = 0.8,
        class_fraction: float = 0.8,
        min_cloud_points: int = 20,
        max_query_points: int = 500,
        max_class_points: int = 1000,
        small_cloud_threshold: int = 100,
        subsample_mode: str = "maxmin",
        class_mode: str = "A",
        rep_names: tuple[str, ...] = ("mtd",),
        hom_dims: tuple[int, ...] = (0, 1),
        blocks: tuple[str, ...] = ("b1", "b2", "b3"),
        density_repeats: int = 1,
        density_estimator: str = "kde",
        pdist_device: str = "cuda",
        random_state: int = 42,
        record_timing: bool = False,
        debug_sizes: bool = False,
        cache_dir: str | None = None,
    ):
        self.vectorizer = vectorizer
        self.embedding_dimension = embedding_dimension
        self.embedding_time_delay = embedding_time_delay
        self.search_embedding = search_embedding
        self.stride = stride
        self.n_components = n_components
        # Cross-barcodes are asymmetric, so the left (query) and right (class)
        # clouds are sized independently. The left is a single series: a fraction
        # of that series' embedded points. The right is a fraction of the pooled
        # class cloud (which can be far larger), with its own higher cap. Sizes
        # are still matched between self-density / train / test per class
        # (methodology §8) because the same target M' is used in fit and transform.
        self.per_series_fraction = per_series_fraction
        self.query_fraction = query_fraction
        self.class_fraction = class_fraction
        self.min_cloud_points = min_cloud_points
        self.max_query_points = max_query_points
        self.max_class_points = max_class_points
        self.small_cloud_threshold = small_cloud_threshold
        self.subsample_mode = subsample_mode
        self.class_mode = class_mode
        self.rep_names = rep_names
        self.hom_dims = hom_dims
        self.blocks = blocks
        self.density_repeats = density_repeats
        self.density_estimator = density_estimator
        self.pdist_device = pdist_device
        self.random_state = random_state
        self.record_timing = record_timing
        self.debug_sizes = debug_sizes
        self.cache_dir = cache_dir

    def fit(self, X, y):
        """Leave-one-series-out pass: builds class clouds, freezes self-densities,
        and caches the leakage-free train feature matrix (see methodology §8-9)."""
        fit_start = time.perf_counter()
        timings: dict[str, float] = {}

        X = check_array(X, dtype=float, ensure_all_finite=True)
        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self.rng_ = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        self.n_train_samples_ = X.shape[0]

        t0 = time.perf_counter()
        self._fit_embedder(X)
        timings["embedding_params"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        embedded_train = [embed_series(X[idx], self.embedder_) for idx in range(X.shape[0])]
        stacked = np.vstack(embedded_train)
        self.pca_ = PCA(n_components=min(self.n_components, stacked.shape[1]))
        self.pca_.fit(stacked)
        timings["pca_fit"] = time.perf_counter() - t0

        self._resolve_sizes([e.shape[0] for e in embedded_train])

        # Pooled class cloud (with provenance) per class. Provenance == train row
        # index, which is what leakage exclusion keys on (methodology §9).
        t0 = time.perf_counter()
        self.class_clouds_: dict[int, np.ndarray] = {}
        self.class_provenance_: dict[int, np.ndarray] = {}
        for class_label in self.classes_:
            class_indices = np.where(y == class_label)[0]
            series_list = [X[idx] for idx in class_indices]
            cloud, prov = build_class_cloud(
                series_list,
                source_indices=class_indices.tolist(),
                embedder=self.embedder_,
                subsample_mode=self.subsample_mode,
                pca=self.pca_,
                fraction=self.per_series_fraction,
                min_points=self.min_cloud_points,
                max_points=self.max_query_points,
                small_threshold=self.small_cloud_threshold,
                rng=self.rng_,
            )
            self.class_clouds_[class_label] = cloud
            self.class_provenance_[class_label] = prov
        # Right (class) cloud size per class: a fraction of the pooled cloud.
        self._resolve_class_sizes()
        # Default (no-exclusion) right cloud per class; deterministic and reused.
        self.class_subclouds_: dict[int, np.ndarray] = {
            class_label: self._class_subcloud(class_label, None)
            for class_label in self.classes_
        }
        timings["class_clouds"] = time.perf_counter() - t0
        self._debug_fit_clouds()

        # --- LOO pass 1: query-vs-class cross-barcodes (the GPU-heavy step) ------
        t0 = time.perf_counter()
        loo_pairs: list[dict[int, tuple]] = []
        loo_extra_qc: list[list] = []
        lifetimes_by_hom: dict[int, list[np.ndarray]] = {h: [] for h in self.hom_dims}
        for i in range(X.shape[0]):
            query_cloud = self._query_cloud(X[i])
            pairs: dict[int, tuple] = {}
            extra_qc: list = []
            for class_label in self.classes_:
                in_class = class_label == y[i]
                right = (
                    self._class_subcloud(class_label, int(i))
                    if in_class
                    else self.class_subclouds_[class_label]
                )
                if self.debug_sizes and i == 0:
                    self._debug_scoring_pair(
                        stage="fit_LOO",
                        row=i,
                        y_i=int(y[i]),
                        class_label=int(class_label),
                        in_class=in_class,
                        query_cloud=query_cloud,
                        right_cloud=right,
                        exclude=None if not in_class else int(i),
                    )
                bc_qc, bc_cq = self._both_orders(query_cloud, right)
                pairs[class_label] = (bc_qc, bc_cq)
                for hom_dim in self.hom_dims:
                    lifetimes_by_hom[hom_dim].append(barcode_lifetimes(bc_qc, hom_dim))
                    lifetimes_by_hom[hom_dim].append(barcode_lifetimes(bc_cq, hom_dim))
                if in_class:
                    for r in range(1, self.density_repeats):
                        right_r = self._class_subcloud(class_label, int(i), repeat=r)
                        bc_qc_r = cross_barcode(
                            query_cloud, right_r, query_cloud.shape[0], right_r.shape[0],
                            self.pdist_device,
                            cache_dir=self.cache_dir,
                        )
                        extra_qc.append((class_label, bc_qc_r))
                        for hom_dim in self.hom_dims:
                            lifetimes_by_hom[hom_dim].append(barcode_lifetimes(bc_qc_r, hom_dim))
            loo_pairs.append(pairs)
            loo_extra_qc.append(extra_qc)
        timings["cross_persistence"] = time.perf_counter() - t0

        # Freeze quantile Betti thresholds from all observed training lifetimes.
        self.betti_thresholds_ = betti_thresholds_from_lifetimes(
            {h: np.concatenate(v) if v else np.zeros(0) for h, v in lifetimes_by_hom.items()}
        )

        # --- LOO pass 2: in-class values are the self-density samples (left side
        # is now a single query series, matching test time) -------------------
        t0 = time.perf_counter()
        collected: dict[int, dict[tuple[str, int], list[float]]] = {
            class_label: {
                (rep_name, hom_dim): []
                for rep_name in self.rep_names
                for hom_dim in self.hom_dims
            }
            for class_label in self.classes_
        }
        for i in range(X.shape[0]):
            in_class = y[i]
            bc_qc = loo_pairs[i][in_class][0]
            self._collect_density(collected[in_class], bc_qc)
            for class_label, bc_qc_r in loo_extra_qc[i]:
                self._collect_density(collected[class_label], bc_qc_r)
        self.class_densities_ = {
            class_label: {
                key: self_density_fit(np.asarray(values, dtype=float))
                for key, values in per_rep.items()
            }
            for class_label, per_rep in collected.items()
        }
        timings["self_densities"] = time.perf_counter() - t0

        # Cached leakage-free train feature matrix (B3 needs the frozen density).
        rows = []
        for i in range(X.shape[0]):
            row_features = []
            for class_label in self.classes_:
                bc_qc, bc_cq = loo_pairs[i][class_label]
                row_features.append(
                    assemble_blocks(
                        bc_qc,
                        bc_cq,
                        self.class_densities_[class_label],
                        self.rep_names,
                        self.hom_dims,
                        self.blocks,
                        betti_thresholds=self.betti_thresholds_,
                        density_estimator=self.density_estimator,
                    )
                )
            rows.append(np.concatenate(row_features))
        self.train_features_ = np.nan_to_num(
            np.vstack(rows), posinf=1e6, neginf=-1e6, nan=0.0
        )

        if self.debug_sizes:
            self._debug(
                f"fit done: train_features_ {self.train_features_.shape}, "
                f"features/class={self._per_class_feature_count()}, "
                f"density_repeats={self.density_repeats}"
            )

        timings["total"] = time.perf_counter() - fit_start
        if self.record_timing:
            self.fit_timings_ = timings
        self.n_features_out_ = len(self.classes_) * self._per_class_feature_count()
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Return the leakage-free train features cached by ``fit`` (no refit pass)."""
        if y is None:
            raise ValueError("TopGenTransformer.fit_transform requires y (class labels)")
        self.fit(X, y)
        return self.train_features_

    def transform(self, X, y=None):
        """Out-of-sample scoring only: single-series query vs deterministic class
        clouds, B3 from frozen self-densities. Use ``fit_transform`` for train rows."""
        transform_start = time.perf_counter()
        query_cloud_time = 0.0
        cross_persistence_time = 0.0

        check_is_fitted(self, "class_clouds_")
        X = check_array(X, dtype=float, ensure_all_finite=True)
        rows = []
        for row_idx in range(X.shape[0]):
            t0 = time.perf_counter()
            query_cloud = self._query_cloud(X[row_idx])
            query_cloud_time += time.perf_counter() - t0
            row_features = []
            for class_label in self.classes_:
                class_subcloud = self.class_subclouds_[class_label]
                if self.debug_sizes and row_idx == 0:
                    self._debug_scoring_pair(
                        stage="transform",
                        row=row_idx,
                        y_i=None,
                        class_label=int(class_label),
                        in_class=None,
                        query_cloud=query_cloud,
                        right_cloud=class_subcloud,
                        exclude=None,
                    )
                t0 = time.perf_counter()
                row_features.append(
                    feature_blocks(
                        query_cloud,
                        class_subcloud,
                        self.class_densities_[class_label],
                        self.rep_names,
                        self.hom_dims,
                        self.blocks,
                        query_cloud.shape[0],
                        class_subcloud.shape[0],
                        pdist_device=self.pdist_device,
                        density_estimator=self.density_estimator,
                        betti_thresholds=self.betti_thresholds_,
                        cache_dir=self.cache_dir,
                    )
                )
                cross_persistence_time += time.perf_counter() - t0
            rows.append(np.concatenate(row_features))
        features = np.vstack(rows)
        features = np.nan_to_num(features, posinf=1e6, neginf=-1e6, nan=0.0)
        if self.record_timing:
            self.transform_timings_ = {
                "query_clouds": query_cloud_time,
                "cross_persistence": cross_persistence_time,
                "total": time.perf_counter() - transform_start,
            }
        return features

    def _seed(self, *parts: object) -> int:
        """Deterministic 32-bit seed derived from random_state and the given parts.

        String parts are folded in via CRC32 so seeds are stable across processes.
        """
        ints = [int(self.random_state) & 0xFFFFFFFF]
        for part in parts:
            if part is None:
                ints.append(0xFFFFFFFF)
            elif isinstance(part, str):
                ints.append(zlib.crc32(part.encode()))
            else:
                ints.append(int(part) & 0xFFFFFFFF)
        return int(np.random.SeedSequence(ints).generate_state(1)[0])

    def _query_cloud(self, series: np.ndarray) -> np.ndarray:
        """Single-series query cloud, deterministic for identical input series."""
        return build_series_cloud(
            series,
            self.embedder_,
            self.subsample_mode,
            self.pca_,
            fraction=self.query_fraction,
            min_points=self.min_cloud_points,
            max_points=self.max_query_points,
            small_threshold=self.small_cloud_threshold,
            seed=self._seed("query"),
        )

    def _class_subcloud(
        self, class_label: int, exclude_series: int | None, repeat: int = 0
    ) -> np.ndarray:
        """Deterministic right cloud C_c' for a (class, exclusion); reused everywhere."""
        seed = self._seed("subcloud", int(class_label), exclude_series, repeat)
        return sample_subcloud(
            self.class_clouds_[class_label],
            self.class_provenance_[class_label],
            self.class_subcloud_size_[class_label],
            self.subsample_mode,
            seed=seed,
            exclude_series=exclude_series,
            class_mode=self.class_mode,
        )

    def _both_orders(self, query_cloud: np.ndarray, right: np.ndarray) -> tuple:
        """Cross-barcodes in both query/class orders."""
        bc_qc = cross_barcode(
            query_cloud, right, query_cloud.shape[0], right.shape[0],
            self.pdist_device, cache_dir=self.cache_dir,
        )
        bc_cq = cross_barcode(
            right, query_cloud, right.shape[0], query_cloud.shape[0],
            self.pdist_device, cache_dir=self.cache_dir,
        )
        return bc_qc, bc_cq

    def _collect_density(self, store: dict[tuple[str, int], list], barcode) -> None:
        """Append per-(rep, hom) scalar values from one query-on-left barcode."""
        for hom_dim in self.hom_dims:
            for rep_name in self.rep_names:
                store[(rep_name, hom_dim)].append(
                    scalar_rep_value(barcode, rep_name, hom_dim, self.betti_thresholds_)
                )

    def _resolve_sizes(self, embedded_counts: list[int]) -> None:
        """Store median embedded count (reference for debug / H1 warning)."""
        ref = int(np.median(embedded_counts)) if embedded_counts else self.min_cloud_points
        self.n_embed_ref_ = ref
        self._example_query_budget_ = cloud_budget(
            ref,
            self.query_fraction,
            self.min_cloud_points,
            self.max_query_points,
            self.small_cloud_threshold,
        )
        self._example_per_series_budget_ = cloud_budget(
            ref,
            self.per_series_fraction,
            self.min_cloud_points,
            self.max_query_points,
            self.small_cloud_threshold,
        )
        if self.debug_sizes:
            rule = (
                "all points (< threshold)"
                if ref < self.small_cloud_threshold
                else f"fraction (threshold={self.small_cloud_threshold})"
            )
            self._debug(
                f"embedding ref: median_embedded={ref} ({rule}); "
                f"example per_series_budget={self._example_per_series_budget_}, "
                f"example query_budget={self._example_query_budget_}"
            )

    def _resolve_class_sizes(self) -> None:
        """Right-side budget per class: fraction of pooled cloud (all if small)."""
        self.class_subcloud_size_: dict[int, int] = {}
        for class_label in self.classes_:
            cloud = self.class_clouds_[class_label]
            prov = self.class_provenance_[class_label]
            n_total = cloud.shape[0]
            if n_total < self.small_cloud_threshold:
                target = n_total
            else:
                target = int(
                    np.clip(
                        round(self.class_fraction * n_total),
                        self.min_cloud_points,
                        self.max_class_points,
                    )
                )
            if n_total and prov.size:
                max_series_pts = int(np.unique(prov, return_counts=True)[1].max())
                loo_available = n_total - max_series_pts
                if loo_available >= self.min_cloud_points:
                    target = min(target, loo_available)
            self.class_subcloud_size_[class_label] = int(max(1, min(target, n_total)))

        if 1 in self.hom_dims:
            smallest = min(self.class_subcloud_size_.values(), default=0)
            example_q = getattr(self, "_example_query_budget_", self.min_cloud_points)
            if min(example_q, smallest) < 15:
                warnings.warn(
                    f"Resolved cloud sizes (example query={example_q}, "
                    f"min class={smallest}) are small for H1; cross-barcode H1 may be "
                    "noisy. Consider a smaller stride or H0-only hom_dims.",
                    stacklevel=2,
                )

    def _debug(self, message: str) -> None:
        if self.debug_sizes:
            print(f"[TopGen sizes] {message}")

    def _debug_fit_clouds(self) -> None:
        if not self.debug_sizes:
            return
        self._debug(
            f"Takens: tau={self.embedding_time_delay_}, m={self.embedding_dimension_}, "
            f"stride={self.stride_}, PCA dim={self.pca_.n_components_}"
        )
        for class_label in self.classes_:
            cloud = self.class_clouds_[class_label]
            sub = self.class_subclouds_[class_label]
            n_series = len(np.unique(self.class_provenance_[class_label]))
            self._debug(
                f"class {class_label}: pooled |C|={cloud.shape[0]} from {n_series} series, "
                f"target M'={self.class_subcloud_size_[class_label]}, "
                f"actual C'={sub.shape[0]}"
            )

    def _debug_scoring_pair(
        self,
        stage: str,
        row: int,
        y_i: int | None,
        class_label: int,
        in_class: bool | None,
        query_cloud: np.ndarray,
        right_cloud: np.ndarray,
        exclude: int | None,
    ) -> None:
        if not self.debug_sizes:
            return
        tag = f"{stage} row={row}"
        if y_i is not None:
            tag += f" y={y_i}"
        if in_class is not None:
            tag += f" in_class={in_class}"
        if exclude is not None:
            tag += f" exclude_series={exclude}"
        self._debug(
            f"{tag} vs class {class_label}: "
            f"Q {query_cloud.shape}  C' {right_cloud.shape}  →  "
            f"cross_barcode(Q,C') batch=({query_cloud.shape[0]}, {right_cloud.shape[0]}), "
            f"cross_barcode(C',Q) batch=({right_cloud.shape[0]}, {query_cloud.shape[0]})"
        )

    def _fit_embedder(self, X: np.ndarray) -> None:
        """Pick Takens (tau, m): median of a search over several series, then cap so
        the embedding fits every series (extends Topological_classifier's X[0]-only)."""
        X = np.asarray(X, dtype=float)
        lengths = [np.asarray(row).ravel().shape[0] for row in X]
        min_len = int(min(lengths))
        sample_idx = np.unique(
            np.linspace(0, X.shape[0] - 1, num=min(5, X.shape[0])).astype(int)
        )

        if self.search_embedding:
            time_delay_cap, dimension_cap = _embedding_search_caps(
                min_len, self.embedding_time_delay, self.embedding_dimension, self.stride
            )
            taus, dims = [], []
            for idx in sample_idx:
                search_embedder = SingleTakensEmbedding(
                    parameters_type="search",
                    n_jobs=-1,
                    stride=self.stride,
                    time_delay=time_delay_cap,
                    dimension=dimension_cap,
                )
                search_embedder.fit(np.asarray(X[idx], dtype=float).ravel())
                taus.append(search_embedder.time_delay_)
                dims.append(search_embedder.dimension_)
            time_delay = int(round(float(np.median(taus))))
            dimension = int(round(float(np.median(dims))))
            stride = self.stride
            # Guarantee the median choice still fits the shortest series.
            time_delay, dimension, stride = _cap_fixed_embedding(
                min_len, time_delay, dimension, stride
            )
        else:
            time_delay, dimension, stride = _cap_fixed_embedding(
                min_len,
                self.embedding_time_delay,
                self.embedding_dimension,
                self.stride,
            )
            if (
                time_delay != self.embedding_time_delay
                or dimension != self.embedding_dimension
                or stride != self.stride
            ):
                warnings.warn(
                    f"Short series (min length={min_len}): capped Takens embedding from "
                    f"tau={self.embedding_time_delay}, m={self.embedding_dimension}, "
                    f"stride={self.stride} to tau={time_delay}, m={dimension}, stride={stride}.",
                    stacklevel=2,
                )

        self.embedding_time_delay_ = time_delay
        self.embedding_dimension_ = dimension
        self.stride_ = stride
        self.embedder_ = TakensEmbedding(
            time_delay=time_delay,
            dimension=dimension,
            stride=stride,
        )

    def _per_class_feature_count(self) -> int:
        n_reps = len(self.rep_names)
        n_hom = len(self.hom_dims)
        count = 0
        if "b1" in self.blocks:
            count += 2 * n_reps * n_hom
        if "b2" in self.blocks:
            count += n_reps * n_hom
        if "b3" in self.blocks:
            count += n_reps * n_hom
        return count

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "classes_")
        names = []
        for class_label in self.classes_:
            for hom_dim in self.hom_dims:
                for rep_name in self.rep_names:
                    prefix = f"class_{class_label}_{rep_name}_h{hom_dim}"
                    if "b1" in self.blocks:
                        names.extend([f"{prefix}_qc", f"{prefix}_cq"])
                    if "b2" in self.blocks:
                        names.append(f"{prefix}_asym")
                    if "b3" in self.blocks:
                        names.append(f"{prefix}_membership")
        return np.asarray(names, dtype=object)
