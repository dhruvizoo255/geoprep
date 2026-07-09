"""Tests for geoprep.alignment.service -- optical/SAR alignment.

Includes a regression test for a bug where an empty sar_scenes list
caused every optical scene to be silently dropped instead of producing
optical-only observations.
"""

from __future__ import annotations

import numpy as np
import pytest

from geoprep.alignment.service import AlignmentService
from geoprep.core.models import RasterScene


def make_scene(scene_id: str, modality: str, date: str) -> RasterScene:
    rng = np.random.default_rng(0)
    h, w = 8, 8
    bands = {"VV": (rng.random((h, w))).astype(np.float32)} if modality == "sar" else {
        "red": (rng.random((h, w))).astype(np.float32),
        "nir": (rng.random((h, w))).astype(np.float32),
    }
    return RasterScene(scene_id=scene_id, modality=modality, acquisition_date=date, crs="EPSG:4326", resolution_m=10.0, bands=bands)


def test_optical_only_scenes_are_not_dropped():
    """Regression test: previously, sar_scenes=[] caused aligned == [] always."""

    optical = [make_scene("S2_a", "optical", "2026-01-01")]
    aligned = AlignmentService().align(optical_scenes=optical, sar_scenes=[])

    assert len(aligned) == 1
    assert aligned[0].optical is not None
    assert aligned[0].sar is None


def test_sar_only_scenes_are_not_dropped():
    sar = [make_scene("S1_a", "sar", "2026-01-01")]
    aligned = AlignmentService().align(optical_scenes=[], sar_scenes=sar)

    assert len(aligned) == 1
    assert aligned[0].optical is None
    assert aligned[0].sar is not None


def test_optical_and_sar_within_window_are_paired():
    optical = [make_scene("S2_a", "optical", "2026-01-01")]
    sar = [make_scene("S1_a", "sar", "2026-01-02")]
    aligned = AlignmentService(temporal_window_days=3).align(optical_scenes=optical, sar_scenes=sar)

    assert len(aligned) == 1
    assert aligned[0].optical is not None
    assert aligned[0].sar is not None


def test_optical_and_sar_outside_window_become_single_modality():
    optical = [make_scene("S2_a", "optical", "2026-01-01")]
    sar = [make_scene("S1_a", "sar", "2026-02-01")]
    aligned = AlignmentService(temporal_window_days=3).align(optical_scenes=optical, sar_scenes=sar)

    assert len(aligned) == 2
    pairings = {obs.metadata.get("modality_pairing") for obs in aligned}
    assert pairings == {"optical_only", "sar_only"}


def test_empty_inputs_produce_empty_output():
    aligned = AlignmentService().align(optical_scenes=[], sar_scenes=[])
    assert aligned == []
