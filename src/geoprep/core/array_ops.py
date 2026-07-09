"""
GeoPrep v2
High-performance numerical backend.

All operations are NumPy-vectorized and intended for
large satellite rasters (e.g. full-resolution Sentinel-2 tiles,
10980x10980 pixels). No function in this module iterates over
individual pixels in Python; every operation is delegated to
NumPy, OpenCV, or SciPy's compiled kernels.

Author:
GeoPrep v2
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "opencv-python is required for geoprep.core.array_ops "
        "(resampling and Gaussian filtering)."
    ) from exc

from geoprep.core.logging import get_logger

logger = get_logger("geoprep.core.array_ops")

Array2D = np.ndarray


# ==========================================================
# Validation
# ==========================================================


def ensure_same_shape(*arrays: Array2D) -> tuple[int, int]:
    """
    Ensure all arrays share the same shape.

    Parameters
    ----------
    *arrays : Array2D
        One or more NumPy arrays to compare.

    Returns
    -------
    tuple[int, int]
        The shared shape.

    Raises
    ------
    ValueError
        If no arrays are supplied, or if any array's shape differs
        from the first array's shape.
    """

    if len(arrays) == 0:
        raise ValueError("No arrays supplied.")

    shape = arrays[0].shape

    for arr in arrays:
        if arr.shape != shape:
            raise ValueError(f"Shape mismatch {arr.shape} != {shape}")

    return shape


# ==========================================================
# Basic Operations
# ==========================================================


def scale(
    array: Array2D,
    factor: float,
    offset: float = 0.0,
) -> Array2D:
    """
    Linear scaling.

    y = x*factor + offset

    Vectorized over the full array; no Python-level iteration.
    """

    return array.astype(np.float32) * factor + offset


def clip(
    array: Array2D,
    minimum: float,
    maximum: float,
) -> Array2D:
    """Clip array values to [minimum, maximum], vectorized via numpy.clip."""

    return np.clip(array, minimum, maximum)


def flatten(
    array: Array2D,
) -> np.ndarray:
    """Flatten an array to 1D using NumPy's contiguous ravel (no copy when possible)."""

    return array.ravel()


# ==========================================================
# Ratios
# ==========================================================


def safe_ratio(
    numerator: Array2D,
    denominator: Array2D,
    eps: float = 1e-6,
) -> Array2D:
    """
    Elementwise numerator / denominator, guarding against division by
    (near-)zero by flooring the denominator's magnitude at `eps`.
    """

    ensure_same_shape(numerator, denominator)

    denominator = denominator.astype(np.float32, copy=True)
    small = np.abs(denominator) < eps
    denominator[small] = np.where(denominator[small] >= 0, eps, -eps)

    return numerator / denominator


def normalized_difference(
    a: Array2D,
    b: Array2D,
    eps: float = 1e-6,
) -> Array2D:
    """
    Elementwise (a - b) / (a + b), guarding the denominator against
    (near-)zero. Used for NDVI, NDWI, and similar spectral indices.
    """

    ensure_same_shape(a, b)

    denominator = (a + b).astype(np.float32, copy=True)
    small = np.abs(denominator) < eps
    denominator[small] = np.where(denominator[small] >= 0, eps, -eps)

    return (a - b) / denominator


# ==========================================================
# Resampling
# ==========================================================


def nearest_resample(
    array: Array2D,
    target_rows: int,
    target_cols: int,
) -> Array2D:
    """
    Fast nearest-neighbour resize using OpenCV's compiled resize kernel.

    Used for deterministic raster alignment (e.g. masks, categorical
    layers) where interpolation between classes would be invalid.
    """

    if target_rows <= 0 or target_cols <= 0:
        raise ValueError("Target dimensions must be positive.")

    return cv2.resize(
        np.asarray(array, dtype=np.float32),
        (target_cols, target_rows),
        interpolation=cv2.INTER_NEAREST,
    )


def bilinear_resample(
    array: Array2D,
    target_rows: int,
    target_cols: int,
) -> Array2D:
    """
    Bilinear interpolation resize using OpenCV.

    Recommended for continuous optical/SAR imagery where smooth
    interpolation between neighbouring pixels is appropriate.
    """

    if target_rows <= 0 or target_cols <= 0:
        raise ValueError("Target dimensions must be positive.")

    return cv2.resize(
        np.asarray(array, dtype=np.float32),
        (target_cols, target_rows),
        interpolation=cv2.INTER_LINEAR,
    )


# ==========================================================
# Filters
# ==========================================================


def mean_filter(
    array: Array2D,
    radius: int = 1,
) -> Array2D:
    """
    Fast mean filter via SciPy's compiled uniform_filter.

    radius=1 equals a 3x3 window.
    """

    if radius < 0:
        raise ValueError("radius must be non-negative.")

    kernel = radius * 2 + 1

    return uniform_filter(
        np.asarray(array, dtype=np.float32),
        size=kernel,
        mode="nearest",
    )


def gaussian_filter(
    array: Array2D,
    kernel_size: int = 5,
) -> Array2D:
    """
    Gaussian smoothing via OpenCV's compiled GaussianBlur.

    kernel_size must be a positive odd integer.
    """

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    return cv2.GaussianBlur(
        np.asarray(array, dtype=np.float32),
        (kernel_size, kernel_size),
        sigmaX=0,
    )


# ==========================================================
# Normalization
# ==========================================================


def minmax_normalize(
    array: Array2D,
) -> Array2D:
    """Scale array values to [0, 1] using the array's own min/max."""

    minimum = array.min()
    maximum = array.max()

    if maximum == minimum:
        return np.zeros_like(array, dtype=np.float32)

    return (array - minimum) / (maximum - minimum)


def zscore_normalize(
    array: Array2D,
) -> Array2D:
    """Standardize array values to zero mean and unit variance."""

    mean = array.mean()
    std = array.std()

    if std < 1e-8:
        std = 1.0

    return (array - mean) / std


# ==========================================================
# Statistics
# ==========================================================


def standardize(
    values: np.ndarray,
    mean_value: float | None = None,
    std_value: float | None = None,
) -> np.ndarray:
    """
    Standardize a 1D (or ND) array of values, optionally against a
    supplied mean/std (e.g. train-split statistics applied to val/test).
    """

    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return values

    mu = values.mean() if mean_value is None else mean_value
    sigma = values.std() if std_value is None else std_value

    if sigma < 1e-8:
        sigma = 1.0

    return (values - mu) / sigma


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Cosine similarity between two 1D vectors. Returns 0.0 for degenerate (near-zero) vectors."""

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator < 1e-8:
        return 0.0

    return float(np.dot(a, b) / denominator)


# ==========================================================
# Misc
# ==========================================================


def to_numpy(
    array,
) -> np.ndarray:
    """Coerce any array-like input to a float32 NumPy array."""

    return np.asarray(array, dtype=np.float32)


def zeros_like(
    array: Array2D,
) -> Array2D:
    """Return a float32 zero array with the same shape as `array`."""

    return np.zeros_like(array, dtype=np.float32)


def ones_like(
    array: Array2D,
) -> Array2D:
    """Return a float32 one-filled array with the same shape as `array`."""

    return np.ones_like(array, dtype=np.float32)
