# Architecture

## Design Boundary

GeoPrep's responsibility ends at producing a validated, model-ready dataset on disk. It never trains a model, runs inference, or holds a training loop. This keeps the library composable with any downstream framework (PyTorch, PyTorch Geometric, or otherwise) instead of coupling to one.

## Stage Flow

```
 1. discovery        find candidate scenes for a region/time window
 2. download          (external to this repo) fetch raw scene files
 3. preprocessing     reflectance scaling, cloud masking, radiometric
                       calibration, dB conversion, speckle filtering,
                       resampling
 4. alignment         pair optical + SAR scenes within a temporal window,
                       or pass through single-modality observations
 5. features          NDVI / EVI / NDWI / SAVI / GCI, texture measures,
                       SAR VV/VH/ratio
 6. temporal          fixed-length, gap-filled, masked sequences
 7. graph             k-NN or distance-threshold spatial graphs
 8. validation        CRS/cloud/duplicate/timestamp/graph integrity
                       checks; writes a JSON report
 9. model_inputs      exports normalized ViT/Temporal/GNN tensors with
                       train/val/test splits
```

Each stage is a separate service class with a single public entry point (e.g. `PreprocessingService.preprocess_optical`), so any stage can be called in isolation without running the full pipeline. `GeoPrepPipeline.run_from_scenes` is a convenience orchestrator over stages 3-9 for the common case.

## Why Vectorization Matters Here

A full-resolution Sentinel-2 band is 10,980 × 10,980 pixels — about 120 million pixels per band, times as many bands as are in use. A Python-level loop over that raster (even a "simple" one like `for row in array: for value in row:`) does 120M+ individual Python bytecode operations, which is orders of magnitude slower than the single vectorized instruction NumPy/OpenCV/SciPy would use instead. Every core numerical function in this library is built to operate on the whole array at once for exactly this reason.

## Why the Vision Tensor Export Streams to Disk

Even with vectorized array math, a full tile at 256×256 patches can produce ~1,800 patches per scene. Holding all of them (plus a normalized copy) in memory at once is a different problem than raw compute speed — it's a memory-architecture problem. `model_inputs/service.py` solves it with three bounded passes:

1. **Extract**: slide a window over each feature cube, skip invalid patches (NaN, constant, empty), write each valid patch immediately to a scratch file, and discard it from memory.
2. **Statistics**: compute per-channel mean/std from the train split only, one patch at a time, using Welford's online algorithm — memory cost is O(channels), not O(patches).
3. **Normalize**: re-read, normalize, and persist each patch once, deleting its scratch file as it goes.

At no point does the full patch set exist in memory simultaneously.

## Common Bug Patterns in This Codebase (and How to Avoid Them)

- **NumPy truthiness.** `if some_array:` throws `ValueError: truth value of an array is ambiguous` for any array with more than one element. Several past bugs here (`_cloud_mask`, `_require_band`, `validation/service.py`) came from `or`/`if` chains implicitly relying on truthy checks. Always use `is None` / `is not None`.
- **Dataclass type contracts.** If a model field is typed `np.ndarray`, passing it a plain Python list is a silent contract violation that only breaks downstream (e.g. `edge_index[0] + edge_index[1]` does element-wise addition on real arrays but list concatenation on lists — completely different, and both run without error).
- **Fast but wrong.** A vectorized rewrite of a loop-based function can *look* correct (right shape, no exceptions) while being numerically wrong — see the `_WelfordChannelStats` bug in `v0.2.0`'s changelog entry. Always verify vectorized numerical code against a known-correct reference on real or synthetic data with a known answer, not just "does it run."
