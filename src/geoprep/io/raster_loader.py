"""
GeoPrep Raster Loader

Loads GeoTIFF imagery into the GeoPrep RasterScene format.

Author: GeoPrep
Version: 1.1
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import rasterio

from geoprep.core.models import RasterScene


def load_geotiff(
    path: str | Path,
    modality: str = "optical",
    scene_id: Optional[str] = None,
    acquisition_date: str = "unknown",
) -> RasterScene:
    """
    Load a GeoTIFF into a GeoPrep RasterScene.

    Parameters
    ----------
    path : str | Path
        Path to the GeoTIFF.

    modality : str
        "optical" or "sar"

    scene_id : str, optional
        Scene identifier.

    acquisition_date : str
        Acquisition date (YYYY-MM-DD if known)

    Returns
    -------
    RasterScene
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {path}")

    with rasterio.open(path) as src:

        bands = {}

        # Read every band
        for i in range(1, src.count + 1):
            bands[f"B{i:02d}"] = src.read(i)

        metadata = {
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "transform": src.transform,
            "bounds": src.bounds,
        }

        scene = RasterScene(
            scene_id=scene_id or path.stem,
            modality=modality,
            acquisition_date=acquisition_date,
            crs=str(src.crs),
            resolution_m=float(src.res[0]),
            bands=bands,
            cloud_mask=None,
            metadata=metadata,
        )

    return scene


def describe_scene(scene: RasterScene) -> None:
    """
    Print information about a RasterScene.
    """

    print("=" * 60)
    print("GeoPrep Raster Summary")
    print("=" * 60)

    print(f"Scene ID        : {scene.scene_id}")
    print(f"Modality        : {scene.modality}")
    print(f"Date            : {scene.acquisition_date}")
    print(f"CRS             : {scene.crs}")
    print(f"Resolution      : {scene.resolution_m} m")
    print(f"Band Count      : {len(scene.bands)}")

    print("\nBands")

    for name, band in scene.bands.items():
        print(
            f"  {name:<5}"
            f" Shape={band.shape}"
            f" Type={band.dtype}"
        )

    print("=" * 60)