# GeoPrep

GeoPrep is a modular geospatial preprocessing library that turns raw Sentinel-1 (SAR) and Sentinel-2 (optical) imagery into AI-ready tensors for **Vision Transformers**, **Temporal Transformers**, and **Graph Neural Networks**.

GeoPrep does **not** train or run models. Its job stops the moment a clean, validated, model-ready dataset exists on disk, what happens after that (training, inference, serving) is deliberately out of scope, so the library stays a single-responsibility preprocessing tool that can sit in front of any modeling stack.

## Why GeoPrep

Real Sentinel-2 tiles are large (10,980 × 10,980 pixels per band). Every numerical stage in GeoPrep is built around that constraint:

- All raster operations are **vectorized** with NumPy, OpenCV, and SciPy — no per-pixel Python loops anywhere in the processing path.
- Vision Transformer tensors are extracted and normalized as **patches streamed to disk** in three bounded passes (extract → train-split statistics via Welford's online algorithm → normalize), so a full tile's patch set is never held in memory at once.
- Optical-only, SAR-only, and paired optical+SAR workflows are all first-class — you don't need both modalities to run the pipeline.

## Pipeline Stages

| Stage | Module | Responsibility |
|---|---|---|
| 1 | `discovery` | Find candidate satellite/weather/ancillary datasets for a region and time window |
| 2 | *(download, external)* | Fetch raw scenes from a provider |
| 3 | `geoprep.preprocessing` | Reflectance scaling, cloud masking, radiometric calibration, dB conversion, speckle filtering, resampling |
| 4 | `geoprep.alignment` | Temporal, CRS, and grid alignment between optical and SAR scenes (optical-only / SAR-only observations are preserved, not dropped) |
| 5 | `geoprep.features` | NDVI, EVI, NDWI, SAVI, GCI, texture contrast/entropy/homogeneity, SAR VV/VH/ratio |
| 6 | `geoprep.temporal` | Fixed-length, gap-filled, masked sequences for Temporal Transformers |
| 7 | `geoprep.graph` | k-NN or distance-threshold spatial graphs, PyTorch Geometric-compatible (`node_features`, `edge_index`, `edge_attr`) |
| 8 | `geoprep.validation` | CRS consistency, cloud cover, duplicate/timestamp checks, graph integrity — writes a JSON validation report |
| 9 | `geoprep.model_inputs` | Exports normalized, split (train/val/test) tensors for ViT, Temporal Transformer, and GNN training |

## Installation

```bash
git clone https://github.com/<your-username>/geoprep.git
cd geoprep
pip install -e .
```

Or, without an editable install:

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+. Core dependencies: `numpy`, `opencv-python`, `scipy`, `rasterio`, `pyproj`, `shapely`, `geopandas`, `xarray`, `rioxarray`. No TensorFlow dependency; PyTorch/PyTorch Geometric compatibility is maintained on the output side (tensors are plain NumPy arrays that convert directly).

## Quickstart

```python
import numpy as np
from geoprep.core.models import RasterScene
from geoprep.pipeline import GeoPrepPipeline

# A single Sentinel-2 optical scene (bands as float32 NumPy arrays)
scene = RasterScene(
    scene_id="S2_20260301",
    modality="optical",
    acquisition_date="2026-03-01",
    crs="EPSG:32643",
    resolution_m=10.0,
    bands={
        "blue": blue_band,   # np.ndarray, e.g. shape (10980, 10980)
        "green": green_band,
        "red": red_band,
        "nir": nir_band,
    },
)

# Nodes for the spatial graph (e.g. field parcels or weather stations)
graph_nodes = [
    {"x": 74.31, "y": 28.62, "features": [0.4, 0.2, 0.1],
     "irrigation_id": "canal_1", "soil_type": "loam", "rainfall_mm": 120.0},
    {"x": 74.35, "y": 28.60, "features": [0.5, 0.3, 0.2],
     "irrigation_id": "canal_1", "soil_type": "loam", "rainfall_mm": 118.0},
]

pipeline = GeoPrepPipeline()

dataset = pipeline.run_from_scenes(
    optical=[scene],
    sar=[],              # optional -- optical-only workflows are fully supported
    graph_nodes=graph_nodes,
)

print(dataset.vision_transformer["shape"])       # e.g. [num_patches, channels, 256, 256]
print(dataset.vision_transformer["patch_dir"])   # where patches were streamed to disk
print(dataset.metadata["graph_statistics"])
```

See `examples/quickstart.py` for a runnable version of this script, and `notebooks/` for an end-to-end smoke test.

You can also call each stage directly instead of `run_from_scenes` if you only need part of the pipeline:

```python
clean = pipeline.preprocessing.preprocess_optical(scene, scene.crs, scene.resolution_m, scene.band_shape())
aligned = pipeline.alignment.align([clean], [])
cubes = pipeline.features.generate(aligned)
```

## Project Structure

```
geoprep/
├── src/
│   ├── geoprep/            # core preprocessing package (see table above)
│   │   ├── core/           # shared models, exceptions, array_ops, logging, io
│   │   ├── preprocessing/
│   │   ├── alignment/
│   │   ├── features/
│   │   ├── temporal/
│   │   ├── graph/
│   │   ├── validation/
│   │   ├── model_inputs/
│   │   └── pipeline.py     # orchestrates stages 3-9
│   └── discovery/          # stage 1: dataset discovery
├── tests/                  # pytest suite
├── examples/                # runnable example scripts
├── notebooks/               # exploratory / smoke-test notebooks
├── docs/                    # architecture notes
├── pyproject.toml
└── requirements.txt
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Status

Active development. See `CHANGELOG.md` for release notes.

## Contributing

Contributions are welcome — see `CONTRIBUTING.md`.

## License

MIT — see `LICENSE`.
