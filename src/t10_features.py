"""T10 · Domain Shift Radar — label-free descriptors and drift statistics.

Everything here uses only inputs a deployed system would have (images + model scores),
never ground truth. Statistics are calibrated with *permutation tests*: pool reference and
batch, shuffle the batch labels, recompute the statistic. The observed statistic is turned
into a z-score against that null, so "how surprising is this batch?" is comparable across
descriptors of different scale.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Descriptors that describe image *conditions* (not content). Order matters for reports only.
DESCRIPTORS = [
    "brightness",     # mean gray level
    "contrast",       # std of gray level
    "log_sharpness",  # log1p(Laplacian variance): blur / noise
    "edge_density",   # fraction of Canny edge pixels: small-object visibility proxy
    "saturation",     # mean HSV saturation
    "color_cast",     # mean(R) - mean(B): tint
    "dark_frac",      # fraction of pixels with gray < 40
    "sky_ground",     # mean(top third) - mean(bottom third) brightness: lighting layout
]


def image_descriptors(path: str | Path) -> dict:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = gray.shape[0]
    edges = cv2.Canny(gray, 50, 150)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "log_sharpness": float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())),
        "edge_density": float((edges > 0).mean()),
        "saturation": float(hsv[..., 1].mean()),
        "color_cast": float(bgr[..., 2].mean() - bgr[..., 0].mean()),
        "dark_frac": float((gray < 40).mean()),
        "sky_ground": float(gray[: h // 3].mean() - gray[2 * h // 3:].mean()),
    }


def uncertainty(p: np.ndarray | pd.Series) -> np.ndarray:
    """1 at p=0.5, 0 at p in {0,1}. Binary-entropy-like, but linear and easy to explain."""
    return 1.0 - 2.0 * np.abs(np.asarray(p, dtype=float) - 0.5)


# ------------------------------------------------------------------ statistics
def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    return float(ks_2samp(a, b).statistic)


def _rbf_mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Unbiased MMD^2 with an RBF kernel (Gretton et al. 2012)."""
    def k(a, b):
        d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d)
    m, n = len(x), len(y)
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1)) - 2 * kxy.mean())


def median_gamma(x: np.ndarray) -> float:
    """Median heuristic bandwidth, computed on the reference only (frozen per run)."""
    d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    med = np.median(d[np.triu_indices(len(x), 1)])
    return 1.0 / max(med, 1e-9)


def permutation_z(stat_fn, ref: np.ndarray, new: np.ndarray, rng: np.random.Generator, n_perm: int = 300) -> dict:
    """Observed statistic, its permutation null mean/std, z-score and one-sided p-value."""
    obs = stat_fn(ref, new)
    pooled = np.concatenate([ref, new])
    m = len(ref)
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(pooled))
        null[i] = stat_fn(pooled[idx[:m]], pooled[idx[m:]])
    sd = null.std(ddof=1) or 1e-9
    return {
        "stat": float(obs),
        "null_mean": float(null.mean()),
        "null_sd": float(sd),
        "z": float((obs - null.mean()) / sd),
        "p": float((1 + (null >= obs).sum()) / (n_perm + 1)),
    }


def standardize(ref: pd.DataFrame, df: pd.DataFrame, cols=DESCRIPTORS) -> np.ndarray:
    """Standardize with reference mean/std only (no information from the new batch leaks in)."""
    mu = ref[cols].mean()
    sd = ref[cols].std(ddof=1).replace(0, 1.0)
    return ((df[cols] - mu) / sd).to_numpy(dtype=float)


def per_descriptor_shift(ref: pd.DataFrame, new: pd.DataFrame, cols=DESCRIPTORS) -> dict:
    """Standardized mean difference per descriptor — used to *explain* a shift, not to score it."""
    sd = ref[cols].std(ddof=1).replace(0, 1.0)
    return ((new[cols].mean() - ref[cols].mean()) / sd).round(2).to_dict()


def knn_atypicality(ref_z: np.ndarray, new_z: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean distance to the k nearest reference images (standardized descriptor space)."""
    d = np.sqrt(((new_z[:, None, :] - ref_z[None, :, :]) ** 2).sum(-1))
    d.sort(axis=1)
    return d[:, :k].mean(axis=1)
