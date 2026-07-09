"""Tests for geoprep.model_inputs.service -- streaming normalization statistics.

Includes a regression test for a bug where a leftover per-pixel counting
loop caused the running sample count (and therefore mean/std) computed by
_WelfordChannelStats to be silently doubled and numerically wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from geoprep.model_inputs.service import _WelfordChannelStats


def test_welford_matches_ground_truth_across_multiple_batches():
    rng = np.random.default_rng(0)
    batch1 = rng.normal(loc=5.0, scale=2.0, size=(1, 100)).astype(np.float32)
    batch2 = rng.normal(loc=5.0, scale=2.0, size=(1, 150)).astype(np.float32)
    batch3 = rng.normal(loc=5.0, scale=2.0, size=(1, 73)).astype(np.float32)

    all_values = np.concatenate([batch1[0], batch2[0], batch3[0]])
    true_mean, true_std = all_values.mean(), all_values.std()

    stats = _WelfordChannelStats(num_channels=1)
    stats.update(batch1.reshape(1, 10, 10))
    stats.update(batch2.reshape(1, 10, 15))
    stats.update(batch3.reshape(1, 73, 1))
    means, stds = stats.finalize()

    assert stats._count == 100 + 150 + 73
    np.testing.assert_allclose(means.ravel(), [true_mean], rtol=1e-4)
    np.testing.assert_allclose(stds.ravel(), [true_std], rtol=1e-4)


def test_welford_single_batch_matches_numpy_directly():
    rng = np.random.default_rng(1)
    patch = rng.normal(loc=1.0, scale=0.5, size=(2, 16, 16)).astype(np.float32)

    stats = _WelfordChannelStats(num_channels=2)
    stats.update(patch)
    means, stds = stats.finalize()

    expected_mean = patch.reshape(2, -1).mean(axis=1)
    expected_std = patch.reshape(2, -1).std(axis=1)

    np.testing.assert_allclose(means.ravel(), expected_mean, rtol=1e-4)
    np.testing.assert_allclose(stds.ravel(), expected_std, rtol=1e-4)


def test_welford_finalize_without_update_raises():
    stats = _WelfordChannelStats(num_channels=1)
    with pytest.raises(Exception):
        stats.finalize()
