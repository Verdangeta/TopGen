"""Level-1 time-series vectorizers (swappable design axis)."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.decomposition import PCA

from topgen.clouds import delay_embed, embed, estimate_embedding_dimension, estimate_tau


def _persistence_stats_vector(series: np.ndarray, tau: int | None, m: int | None, stride: int) -> np.ndarray:
    """Persistence statistics of the delay embedding (placeholder for full H0/H1 stats)."""
    series = np.asarray(series, dtype=float).ravel()
    tau = estimate_tau(series) if tau is None else tau
    m = estimate_embedding_dimension(series, tau) if m is None else m
    points = embed(series, tau, m, stride=stride)
    pca = PCA(n_components=min(3, points.shape[1]))
    cloud = pca.fit_transform(points)
    # Minimal summary until the full persistence-stats arm is wired in.
    return np.concatenate(
        [
            cloud.mean(axis=0),
            cloud.std(axis=0),
            np.array([cloud.shape[0]], dtype=float),
        ]
    )


def _logsignature_vector(series: np.ndarray, level: int = 2) -> np.ndarray:
    """Log-signature vectorizer (requires iisignature at runtime)."""
    try:
        import iisignature
    except ImportError as exc:
        raise ImportError("logsignature vectorizer requires iisignature") from exc
    series = np.asarray(series, dtype=float).ravel()
    path = PCA(n_components=3).fit_transform(series[:, None])
    return iisignature.sig(path, level)


def _catch22_vector(series: np.ndarray) -> np.ndarray:
    """catch22 features (requires pycatch22 at runtime)."""
    try:
        import pycatch22
    except ImportError as exc:
        raise ImportError("catch22 vectorizer requires pycatch22") from exc
    series = np.asarray(series, dtype=float).ravel()
    features = pycatch22.catch22_all(series)
    return np.asarray(features["values"], dtype=float)


_VECTORIZERS: dict[str, Callable[..., np.ndarray]] = {
    "persistence_stats": _persistence_stats_vector,
    "logsignature": _logsignature_vector,
    "catch22": _catch22_vector,
}


def get_vectorizer(name: str, **kwargs) -> Callable[..., np.ndarray]:
    """Return a named level-1 vectorizer."""
    if name not in _VECTORIZERS:
        raise ValueError(f"Unknown vectorizer '{name}'. Choices: {sorted(_VECTORIZERS)}")
    base = _VECTORIZERS[name]

    def selected(series: np.ndarray) -> np.ndarray:
        return base(series, **kwargs)

    return selected
