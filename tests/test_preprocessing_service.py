"""Tests for geoprep.preprocessing.service -- optical/SAR preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from geoprep.core.exceptions import ProcessingError
from geoprep.core.models import RasterScene
from geoprep.preprocessing.service import PreprocessingService


@pytest.fixture
def optical_scene() -> RasterScene:
    rng = np.random.default_rng(0)
    h, w = 20, 20
    return RasterScene(
        scene_id="S2_test",
        modality="optical",
        acquisition_date="2026-01-01",
        crs="EPSG:4326",
        resolution_m=10.0,
        bands={
            "blue": (rng.random((h, w)) * 10000).astype(np.float32),
            "red": (rng.random((h, w)) * 10000).astype(np.float32),
            "nir": (rng.random((h, w)) * 10000).astype(np.float32),
        },
    )


@pytest.fixture
def sar_scene() -> RasterScene:
    rng = np.random.default_rng(1)
    h, w = 20, 20
    return RasterScene(
        scene_id="S1_test",
        modality="sar",
        acquisition_date="2026-01-01",
        crs="EPSG:4326",
        resolution_m=10.0,
        bands={
            "VV": (rng.random((h, w)) * 2).astype(np.float32),
            "VH": (rng.random((h, w)) * 2).astype(np.float32),
        },
    )


def test_preprocess_optical_produces_bounded_reflectance(optical_scene):
    service = PreprocessingService()
    result = service.preprocess_optical(optical_scene, "EPSG:32643", 10.0)
    for band in result.bands.values():
        assert band.min() >= 0.0
        assert band.max() <= 1.0


def test_preprocess_optical_cloud_mask_is_ndarray(optical_scene):
    service = PreprocessingService()
    result = service.preprocess_optical(optical_scene, "EPSG:32643", 10.0)
    assert isinstance(result.cloud_mask, np.ndarray)
    assert result.cloud_mask.dtype == np.bool_


def test_preprocess_optical_resamples_to_target_shape(optical_scene):
    service = PreprocessingService()
    result = service.preprocess_optical(optical_scene, "EPSG:32643", 10.0, target_shape=(10, 10))
    for band in result.bands.values():
        assert band.shape == (10, 10)
    assert result.cloud_mask.shape == (10, 10)


def test_preprocess_optical_rejects_sar_scene(sar_scene):
    service = PreprocessingService()
    with pytest.raises(ProcessingError):
        service.preprocess_optical(sar_scene, "EPSG:32643", 10.0)


def test_preprocess_sar_converts_to_db_range(sar_scene):
    service = PreprocessingService()
    result = service.preprocess_sar(sar_scene, "EPSG:32643", 10.0)
    # Input was in [0, 2] linear scale; after log10 conversion values should
    # be finite and generally negative-to-small (no NaN/inf from log of 0).
    for band in result.bands.values():
        assert np.all(np.isfinite(band))


def test_preprocess_sar_rejects_optical_scene(optical_scene):
    service = PreprocessingService()
    with pytest.raises(ProcessingError):
        service.preprocess_sar(optical_scene, "EPSG:32643", 10.0)
