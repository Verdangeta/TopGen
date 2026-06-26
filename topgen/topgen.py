"""Scikit-learn compatible TopGen v2 population-level feature generator."""

from __future__ import annotations

import time
import warnings

import numpy as np
from gtda.time_series import SingleTakensEmbedding, TakensEmbedding
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted

from topgen.clouds import (
    build_class_cloud,
    build_series_cloud,
    embed_series,
    sample_subcloud,
)
from topgen.features import (
    ALL_REP_NAMES,
    estimate_class_self_densities,
    feature_blocks,
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


def _ensure_embedding_fits(series_length: int, time_delay: int, dimension: int) -> None:
    if series_length <= time_delay * (dimension - 1) + 1:
        raise ValueError(
            f"Series length {series_length} is too short for Takens embedding with "
            f"time_delay={time_delay} and dimension={dimension}."
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
        per_series_budget: int = 50,
        query_size: int = 30,
        class_subcloud_size: int = 40,
        subsample_mode: str = "maxmin",
        class_mode: str = "A",
        rep_names: tuple[str, ...] = ("mtd",),
        hom_dims: tuple[int, ...] = (0, 1),
        blocks: tuple[str, ...] = ("b1", "b2", "b3"),
        density_samples: int = 40,
        density_estimator: str = "kde",
        pdist_device: str = "cuda",
        random_state: int = 42,
        record_timing: bool = False,
    ):
        self.vectorizer = vectorizer
        self.embedding_dimension = embedding_dimension
        self.embedding_time_delay = embedding_time_delay
        self.search_embedding = search_embedding
        self.stride = stride
        self.n_components = n_components
        self.per_series_budget = per_series_budget
        self.query_size = query_size
        self.class_subcloud_size = class_subcloud_size
        self.subsample_mode = subsample_mode
        self.class_mode = class_mode
        self.rep_names = rep_names
        self.hom_dims = hom_dims
        self.blocks = blocks
        self.density_samples = density_samples
        self.density_estimator = density_estimator
        self.pdist_device = pdist_device
        self.random_state = random_state
        self.record_timing = record_timing

    def fit(self, X, y):
        fit_start = time.perf_counter()
        timings: dict[str, float] = {}

        X = check_array(X, dtype=float, ensure_all_finite=True)
        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self.rng_ = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)

        t0 = time.perf_counter()
        self._fit_embedder(X[0])
        timings["embedding_params"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        embedded_train = [embed_series(X[idx], self.embedder_) for idx in range(X.shape[0])]
        stacked = np.vstack(embedded_train)
        self.pca_ = PCA(n_components=min(self.n_components, stacked.shape[1]))
        self.pca_.fit(stacked)
        timings["pca_fit"] = time.perf_counter() - t0

        self.class_clouds_: dict[int, np.ndarray] = {}
        self.class_provenance_: dict[int, np.ndarray] = {}
        self.class_densities_: dict[int, dict] = {}
        self.n_train_samples_ = X.shape[0]

        cloud_time = 0.0
        density_time = 0.0
        for class_label in self.classes_:
            class_mask = y == class_label
            class_indices = np.where(class_mask)[0]
            series_list = [X[idx] for idx in class_indices]

            t0 = time.perf_counter()
            cloud, prov = build_class_cloud(
                series_list,
                source_indices=class_indices.tolist(),
                embedder=self.embedder_,
                s=self.per_series_budget,
                subsample_mode=self.subsample_mode,
                pca=self.pca_,
                rng=self.rng_,
            )
            cloud_time += time.perf_counter() - t0
            self.class_clouds_[class_label] = cloud
            self.class_provenance_[class_label] = prov

            t0 = time.perf_counter()
            self.class_densities_[class_label] = estimate_class_self_densities(
                cloud,
                prov,
                rep_names=self.rep_names,
                hom_dims=self.hom_dims,
                left_size=self.query_size,
                right_size=self.class_subcloud_size,
                n_samples=self.density_samples,
                subsample_mode=self.subsample_mode,
                rng=self.rng_,
                pdist_device=self.pdist_device,
            )
            density_time += time.perf_counter() - t0

        timings["class_clouds"] = cloud_time
        timings["self_densities"] = density_time
        timings["total"] = time.perf_counter() - fit_start
        if self.record_timing:
            self.fit_timings_ = timings
        self.n_features_out_ = len(self.classes_) * self._per_class_feature_count()
        return self

    def transform(self, X, y=None):
        transform_start = time.perf_counter()
        query_cloud_time = 0.0
        cross_persistence_time = 0.0

        check_is_fitted(self, "class_clouds_")
        X = check_array(X, dtype=float, ensure_all_finite=True)
        y_array = None if y is None else np.asarray(y)
        rows = []
        for row_idx in range(X.shape[0]):
            t0 = time.perf_counter()
            query_cloud = build_series_cloud(
                X[row_idx],
                self.embedder_,
                self.query_size,
                self.subsample_mode,
                self.pca_,
                self.rng_,
            )
            query_cloud_time += time.perf_counter() - t0
            row_features = []
            for class_label in self.classes_:
                exclude_series = None
                if (
                    y_array is not None
                    and X.shape[0] == self.n_train_samples_
                    and y_array[row_idx] == class_label
                ):
                    exclude_series = row_idx
                class_subcloud = sample_subcloud(
                    self.class_clouds_[class_label],
                    self.class_provenance_[class_label],
                    self.class_subcloud_size,
                    self.subsample_mode,
                    self.rng_,
                    exclude_series=exclude_series,
                    class_mode=self.class_mode,
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
                        min(self.query_size, query_cloud.shape[0]),
                        min(self.class_subcloud_size, class_subcloud.shape[0]),
                        pdist_device=self.pdist_device,
                        density_estimator=self.density_estimator,
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

    def _fit_embedder(self, reference_series: np.ndarray) -> None:
        """Match Topological_classifier: search on one series, embed all with TakensEmbedding."""
        reference_series = np.asarray(reference_series, dtype=float).ravel()
        series_length = reference_series.shape[0]
        if self.search_embedding:
            time_delay_cap, dimension_cap = _embedding_search_caps(
                series_length,
                self.embedding_time_delay,
                self.embedding_dimension,
                self.stride,
            )
            search_embedder = SingleTakensEmbedding(
                parameters_type="search",
                n_jobs=-1,
                stride=self.stride,
                time_delay=time_delay_cap,
                dimension=dimension_cap,
            )
            search_embedder.fit(reference_series)
            time_delay = search_embedder.time_delay_
            dimension = search_embedder.dimension_
            stride = self.stride
        else:
            time_delay, dimension, stride = _cap_fixed_embedding(
                series_length,
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
                    f"Short series (length={series_length}): capped Takens embedding from "
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
