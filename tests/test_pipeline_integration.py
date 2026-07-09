"""End-to-end integration test for GeoPrepPipeline.run_from_scenes."""

from __future__ import annotations

import numpy as np
import pytest

from geoprep.core.models import RasterScene
from geoprep.pipeline import GeoPrepPipeline


def make_optical_scene(scene_id: str, date: str, height: int = 300, width: int = 300) -> RasterScene:
    rng = np.random.default_rng(0)
    return RasterScene(
        scene_id=scene_id,
        modality="optical",
        acquisition_date=date,
        crs="EPSG:32643",
        resolution_m=10.0,
        bands={
            "blue": (rng.random((height, width)) * 10000).astype(np.float32),
            "green": (rng.random((height, width)) * 10000).astype(np.float32),
            "red": (rng.random((height, width)) * 10000).astype(np.float32),
            "nir": (rng.random((height, width)) * 10000).astype(np.float32),
        },
    )


def make_sar_scene(scene_id: str, date: str, height: int = 300, width: int = 300) -> RasterScene:
    rng = np.random.default_rng(1)
    return RasterScene(
        scene_id=scene_id,
        modality="sar",
        acquisition_date=date,
        crs="EPSG:32643",
        resolution_m=10.0,
        bands={
            "VV": (rng.random((height, width)) * 2).astype(np.float32),
            "VH": (rng.random((height, width)) * 2).astype(np.float32),
        },
    )


@pytest.fixture
def graph_nodes() -> list[dict[str, object]]:
    return [
        {"x": 1.0, "y": 2.0, "features": [0.1, 0.2], "irrigation_id": "c1", "soil_type": "loam", "rainfall_mm": 100.0},
        {"x": 3.0, "y": 4.0, "features": [0.2, 0.3], "irrigation_id": "c1", "soil_type": "loam", "rainfall_mm": 110.0},
    ]


def test_pipeline_runs_optical_only(tmp_path, monkeypatch, graph_nodes):
    monkeypatch.chdir(tmp_path)
    scene = make_optical_scene("S2_only", "2026-03-01")

    dataset = GeoPrepPipeline().run_from_scenes(optical=[scene], sar=[], graph_nodes=graph_nodes)

    assert dataset.vision_transformer["shape"][0] >= 1
    assert dataset.metadata["graph_statistics"]["node_count"] == 2


def test_pipeline_runs_optical_and_sar_paired(tmp_path, monkeypatch, graph_nodes):
    monkeypatch.chdir(tmp_path)
    optical = make_optical_scene("S2_a", "2026-03-01")
    sar = make_sar_scene("S1_a", "2026-03-02")

    dataset = GeoPrepPipeline().run_from_scenes(optical=[optical], sar=[sar], graph_nodes=graph_nodes)

    assert dataset.vision_transformer["shape"][0] >= 1


def test_pipeline_raises_with_no_scenes(graph_nodes):
    with pytest.raises(ValueError):
        GeoPrepPipeline().run_from_scenes(optical=[], sar=[], graph_nodes=graph_nodes)
