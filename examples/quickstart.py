"""Quickstart: run the full GeoPrep pipeline on a synthetic Sentinel-2 scene.

This uses randomly generated bands so it runs anywhere with no data
download required. Swap in real bands read via rasterio (see
`geoprep.io.raster_loader`) to use it with actual Sentinel-2 imagery.

Run with:
    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

from geoprep.core.models import RasterScene
from geoprep.pipeline import GeoPrepPipeline


def make_synthetic_optical_scene(scene_id: str, height: int = 300, width: int = 300) -> RasterScene:
    """Build a synthetic optical scene large enough to extract at least one 256x256 patch."""

    rng = np.random.default_rng(42)
    return RasterScene(
        scene_id=scene_id,
        modality="optical",
        acquisition_date="2026-03-01",
        crs="EPSG:32643",
        resolution_m=10.0,
        bands={
            "blue": (rng.random((height, width)) * 10000).astype(np.float32),
            "green": (rng.random((height, width)) * 10000).astype(np.float32),
            "red": (rng.random((height, width)) * 10000).astype(np.float32),
            "nir": (rng.random((height, width)) * 10000).astype(np.float32),
        },
    )


def main() -> None:
    scene = make_synthetic_optical_scene("S2_demo")

    graph_nodes = [
        {
            "x": 74.31,
            "y": 28.62,
            "features": [0.4, 0.2, 0.1],
            "irrigation_id": "canal_1",
            "soil_type": "loam",
            "rainfall_mm": 120.0,
        },
        {
            "x": 74.35,
            "y": 28.60,
            "features": [0.5, 0.3, 0.2],
            "irrigation_id": "canal_1",
            "soil_type": "loam",
            "rainfall_mm": 118.0,
        },
    ]

    pipeline = GeoPrepPipeline()

    dataset = pipeline.run_from_scenes(
        optical=[scene],
        sar=[],  # optical-only: no SAR scenes required
        graph_nodes=graph_nodes,
    )

    print("Vision transformer tensor shape:", dataset.vision_transformer["shape"])
    print("Vision patches written to:", dataset.vision_transformer["patch_dir"])
    print("Temporal sequence shape:", np.asarray(dataset.temporal_transformer["sequences"]).shape)
    print("Graph statistics:", dataset.metadata["graph_statistics"])
    print("Dataset splits (sample counts):", {key: len(value) for key, value in dataset.splits.items()})


if __name__ == "__main__":
    main()
