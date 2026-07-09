# Changelog

All notable changes to this project are documented in this file.

## v0.2.0

**Full NumPy vectorization migration.** The numerical backend previously used nested Python lists and per-pixel loops in several modules; this release replaces them with vectorized NumPy/OpenCV/SciPy operations throughout, while keeping public APIs unchanged wherever possible.

### Changed
- `core/array_ops.py`: hardened validation and fixed a sign-flip edge case in `safe_ratio` / `normalized_difference` for small negative denominators.
- `preprocessing/service.py`: `_cloud_mask`, `_resample_mask`, and `_to_db` rewritten from per-pixel Python loops to vectorized NumPy operations.
- `features/service.py`: `_evi`, `_savi`, `_local_contrast`, `_entropy`, `_homogeneity` vectorized; fixed a crash (`ValueError: truth value of an array is ambiguous`) caused by truthy checks on band arrays.
- `alignment/service.py`: optical-only and SAR-only observations are now correctly produced (previously, if either modality's scene list was empty, all scenes were silently dropped instead of producing single-modality observations, even though the data model already supported it).
- `graph/service.py`: replaced an O(N²) pure-Python double loop with vectorized pairwise distance/similarity computation; added optional k-nearest-neighbour graph mode alongside the original distance/threshold graph.
- `temporal/service.py`: sequences and masks are now built as real `np.ndarray` (previously plain Python lists, despite the dataclass declaring `np.ndarray`).
- `validation/service.py`: fixed a crash from truthy-checking a cloud mask array directly; fixed an edge-index validation bug that would silently do element-wise addition instead of concatenation once `edge_index` became a real 2×E array; added a valid empty-graph path for vision-only pipelines.
- `model_inputs/service.py`: redesigned vision-tensor export to stream fixed-size patches to disk across three bounded passes (extract → train-split statistics via Welford's online algorithm → normalize) instead of building one large in-memory array — required for tile-scale rasters where materializing the full patch set would exhaust memory.
- `core/models.py`: added the `GraphDataset.to_dict()` method that `model_inputs/service.py` already relied on but which never existed.

### Fixed
- Corrected a bug in `_WelfordChannelStats.update()` where a leftover per-pixel counting loop caused the running sample count (and therefore mean/std) to be silently doubled and mathematically wrong from the first batch onward.

## v0.1.0

- Initial release.
