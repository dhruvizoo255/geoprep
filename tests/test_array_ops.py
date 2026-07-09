"""Tests for geoprep.core.array_ops -- vectorized numerical primitives."""

from __future__ import annotations

import numpy as np
import pytest

from geoprep.core import array_ops as ao


def test_ensure_same_shape_passes_for_matching_arrays():
    a = np.zeros((4, 4))
    b = np.ones((4, 4))
    assert ao.ensure_same_shape(a, b) == (4, 4)


def test_ensure_same_shape_raises_for_mismatched_arrays():
    a = np.zeros((4, 4))
    b = np.zeros((3, 3))
    with pytest.raises(ValueError):
        ao.ensure_same_shape(a, b)


def test_scale_applies_linear_transform():
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = ao.scale(array, factor=2.0, offset=1.0)
    expected = array * 2.0 + 1.0
    np.testing.assert_allclose(result, expected)


def test_safe_ratio_preserves_sign_for_small_denominator():
    numerator = np.array([1.0], dtype=np.float32)
    denominator = np.array([-1e-9], dtype=np.float32)
    result = ao.safe_ratio(numerator, denominator, eps=1e-6)
    # Sign should be preserved: a small *negative* denominator should not
    # silently flip to a positive result.
    assert result[0] < 0


def test_normalized_difference_matches_manual_formula():
    a = np.array([[5.0, 2.0]], dtype=np.float32)
    b = np.array([[3.0, 1.0]], dtype=np.float32)
    result = ao.normalized_difference(a, b)
    expected = (a - b) / (a + b)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_nearest_resample_changes_shape():
    array = np.random.rand(10, 10).astype(np.float32)
    result = ao.nearest_resample(array, 5, 5)
    assert result.shape == (5, 5)


def test_bilinear_resample_changes_shape():
    array = np.random.rand(10, 10).astype(np.float32)
    result = ao.bilinear_resample(array, 20, 20)
    assert result.shape == (20, 20)


def test_mean_filter_smooths_constant_array():
    array = np.full((5, 5), 3.0, dtype=np.float32)
    result = ao.mean_filter(array, radius=1)
    np.testing.assert_allclose(result, array, rtol=1e-5)


def test_minmax_normalize_bounds_are_zero_and_one():
    array = np.array([[1.0, 5.0], [3.0, 9.0]], dtype=np.float32)
    result = ao.minmax_normalize(array)
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)


def test_minmax_normalize_handles_constant_array():
    array = np.full((3, 3), 7.0, dtype=np.float32)
    result = ao.minmax_normalize(array)
    np.testing.assert_allclose(result, np.zeros((3, 3)))


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert ao.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_identical_vectors_is_one():
    a = np.array([1.0, 2.0, 3.0])
    assert ao.cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_degenerate_vector_returns_zero():
    a = np.zeros(3)
    b = np.array([1.0, 1.0, 1.0])
    assert ao.cosine_similarity(a, b) == 0.0
