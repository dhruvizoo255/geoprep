"""Stage 9: AI-ready dataset export for downstream model training systems.

Architectural redesign (v2 — disk-backed streaming patches)
-------------------------------------------------------------
GeoPrep's job stops at producing a ready-to-train dataset on disk. It is
not a TensorFlow/PyTorch DataLoader, and it must never hold an entire
patch dataset in RAM just to normalize and hand it back as one big array.

For a real Sentinel-2 tile (10980x10980, 8 bands) there can be ~1,849
256x256 patches. Even at ~2 MB per patch, stacking all of them
in-memory (`np.stack`) plus a normalized copy easily reaches 7-8 GB.
That is an architecture problem, not a performance problem, and no
amount of vectorization fixes it -- the fix is to never materialize the
full patch set at once.

This version writes patches straight to disk and only ever keeps one
patch (or a handful of running per-channel statistics) in memory at a
time, via three streaming passes:

  Pass 1 (extract):   slide a window over each FeatureCube's (C, H, W)
                       tensor, skip invalid patches (NaN / constant /
                       empty), and write each valid patch immediately
                       to a raw scratch file. Nothing is kept beyond
                       the current patch.

  Pass 2 (statistics): once the total valid-patch count is known, the
                       train/validation/test split boundaries are
                       computed. Only the *train* patches are re-read
                       (one at a time, from the small scratch files
                       written in pass 1 -- no raster re-slicing) to
                       accumulate per-channel mean/std with Welford's
                       online algorithm. Memory cost is O(channels),
                       not O(patches).

  Pass 3 (normalize):  each raw patch is re-read once, normalized with
                       the pass-2 statistics, written to its final
                       location, and the scratch file is deleted.

Output layout:

    outputs/model_inputs/
        vision/
            patch_000000.npy
            patch_000001.npy
            ...
        metadata.json          <- returned via `output_path`

Temporal Transformer and Graph Neural Network exports are unchanged in
structure -- sequences/masks/timestamps and the graph payload are small
relative to raster patches and are not the source of the memory blowup,
so they continue to be embedded directly in the metadata JSON as before.
"""

from __future__ import annotations

import os
import shutil

import numpy as np

from geoprep.core.exceptions import ModelInputError
from geoprep.core.io import write_json
from geoprep.core.logging import get_logger
from geoprep.core.models import FeatureCube, GraphDataset, SequenceDataset, TensorDataset

DEFAULT_PATCH_SIZE = 256
DEFAULT_STRIDE = 256


class ModelInputPreparer:
    """Create normalized ViT (disk-backed patches), Temporal Transformer, and GNN datasets.

    No training or inference happens here -- this stage only assembles,
    normalizes, and persists tensors ready to be handed to a training loop.
    """

    def __init__(self, logger_name: str = "geoprep.model_inputs") -> None:
        self.logger = get_logger(logger_name)

    def prepare(
        self,
        cubes: list[FeatureCube],
        sequence: SequenceDataset,
        graph: GraphDataset,
        output_path: str = "outputs/model_inputs/ai_ready_dataset.json",
        patch_size: int = DEFAULT_PATCH_SIZE,
        stride: int = DEFAULT_STRIDE,
    ) -> TensorDataset:
        """Normalize features, create splits, and export final AI-ready tensors.

        Public API is unchanged: existing callers using
        ``prepare(cubes, sequence, graph, output_path)`` continue to work.
        ``patch_size`` / ``stride`` are optional, defaulting to 256/256.

        Vision patches are streamed to
        ``<dirname(output_path)>/vision/patch_NNNNNN.npy`` rather than held
        in memory; the returned ``TensorDataset.vision_transformer`` holds
        only lightweight metadata (paths, shape, stats), not raw arrays.
        """

        if not cubes:
            raise ModelInputError("Feature cubes are required for model input preparation")
        if patch_size <= 0 or stride <= 0:
            raise ModelInputError("patch_size and stride must be positive integers")

        self.logger.info(
            "Preparing model inputs",
            extra={"stage": "prepare", "patch_size": patch_size, "stride": stride},
        )

        base_dir = os.path.dirname(output_path) or "."
        vision_dir = os.path.join(base_dir, "vision")

        vision = self._vision_dataset(cubes, vision_dir=vision_dir, patch_size=patch_size, stride=stride)

        temporal = {
            "sequences": sequence.sequences,
            "masks": sequence.masks,
            "timestamps": sequence.timestamps,
            "feature_names": list(sequence.feature_names),
        }
        graph_payload = graph.to_dict()

        sample_count = max(vision["shape"][0], len(sequence.sequences), len(graph.node_features))
        splits = _splits(sample_count)

        dataset = TensorDataset(
            vision_transformer=vision,
            temporal_transformer=temporal,
            graph_neural_network=graph_payload,
            splits=splits,
            metadata={
                "feature_cube_count": len(cubes),
                "sequence_length": int(sequence.sequences.shape[1]) if len(sequence.sequences) else 0,
                "graph_statistics": graph.graph_metadata,
                "vision_patch_extraction": {
                    "patch_size": patch_size,
                    "stride": stride,
                    "total_patches": vision["shape"][0],
                    "skipped_patches": vision["skipped_patches"],
                    "patch_dir": vision["patch_dir"],
                    "source_cubes": len(cubes),
                },
                "boundary": "GeoPrep stops here; no model training or inference performed.",
            },
        )
        write_json(output_path, _json_safe(dataset.to_dict()))
        return dataset

    def _vision_dataset(
        self,
        cubes: list[FeatureCube],
        vision_dir: str,
        patch_size: int,
        stride: int,
    ) -> dict[str, object]:
        """Stream-extract, normalize, and persist patches without ever holding them all in RAM."""

        feature_names = tuple(sorted(cubes[0].features.keys()))
        for cube_index, cube in enumerate(cubes):
            if set(cube.features.keys()) != set(feature_names):
                raise ModelInputError(
                    f"FeatureCube at index {cube_index} has feature set "
                    f"{sorted(cube.features.keys())}, expected {list(feature_names)}"
                )

        os.makedirs(vision_dir, exist_ok=True)
        raw_dir = os.path.join(vision_dir, "_raw_scratch")
        os.makedirs(raw_dir, exist_ok=True)

        try:
            # --- Pass 1: extract valid patches, write each straight to a raw scratch
            # file, and forget it immediately. Peak memory here is one cube's
            # (C, H, W) tensor plus one (C, patch, patch) slice at a time.
            num_channels = len(feature_names)
            patch_shape: tuple[int, int, int] | None = None
            skipped_patches = 0
            global_index = 0

            for cube in cubes:
                chw = _cube_to_chw(cube, feature_names)
                for row, col in _patch_grid(chw.shape, patch_size, stride):
                    patch = chw[:, row : row + patch_size, col : col + patch_size]
                    if _is_invalid_patch(patch):
                        skipped_patches += 1
                        continue
                    if patch_shape is None:
                        patch_shape = patch.shape
                    np.save(_raw_patch_path(raw_dir, global_index), patch.astype(np.float32))
                    global_index += 1
                del chw  # release this cube's raster before moving to the next one

            total_patches = global_index
            if total_patches == 0:
                raise ModelInputError(
                    "No valid patches could be extracted from the provided feature cubes "
                    f"(patch_size={patch_size}, stride={stride}). All candidate patches were "
                    "too small, NaN, constant, or empty."
                )

            split_indices = _splits(total_patches)
            train_index_set = set(split_indices["train"])

            # --- Pass 2: accumulate per-channel mean/std from train-split patches
            # only, one raw file at a time, via Welford's online algorithm.
            # Memory cost is O(num_channels), independent of patch count.
            stats = _WelfordChannelStats(num_channels)
            for idx in sorted(train_index_set):
                train_patch = np.load(_raw_patch_path(raw_dir, idx))
                stats.update(train_patch)
                del train_patch
            means, stds = stats.finalize()

            # --- Pass 3: normalize each raw patch (re-read one at a time),
            # write the final patch, and delete the scratch file immediately.
            patch_paths: list[str] = []
            for idx in range(total_patches):
                raw_path = _raw_patch_path(raw_dir, idx)
                patch = np.load(raw_path)
                normalized = ((patch - means) / stds).astype(np.float32)
                final_path = _final_patch_path(vision_dir, idx)
                np.save(final_path, normalized)
                patch_paths.append(os.path.relpath(final_path, start=vision_dir))
                del patch, normalized
                os.remove(raw_path)

        finally:
            shutil.rmtree(raw_dir, ignore_errors=True)

        assert patch_shape is not None
        c, h, w = patch_shape
        return {
            "feature_names": list(feature_names),
            "patch_dir": vision_dir,
            "patch_paths": patch_paths,
            "shape": [total_patches, c, h, w],
            "patch_size": patch_size,
            "stride": stride,
            "skipped_patches": skipped_patches,
            "normalization": {
                "channel_means": means.reshape(-1).tolist(),
                "channel_stds": stds.reshape(-1).tolist(),
                "computed_from": "train split only",
            },
            "samples": [],  # intentionally empty: samples live on disk, not in memory/JSON
        }


class _WelfordChannelStats:
    """Streaming per-channel mean/variance accumulator (Welford's algorithm).

    Update once per patch of shape (C, H, W). Memory usage is fixed at
    O(C) regardless of how many patches are processed or how large each
    patch is.
    """

    def __init__(self, num_channels: int) -> None:
        self._count = 0
        self._mean = np.zeros(num_channels, dtype=np.float64)
        self._m2 = np.zeros(num_channels, dtype=np.float64)

    def update(self, patch: np.ndarray) -> None:
        # Treat every pixel in every channel of this patch as an observation
        # of that channel's distribution. Fully vectorized: no per-pixel
        # Python loop, and self._count is only ever updated once per call
        # (the previous version incremented it via a stray pixel loop AND
        # via new_count below, silently double-counting every patch and
        # producing incorrect running mean/std).
        channels = patch.shape[0]
        flat_per_channel = patch.reshape(channels, -1).astype(np.float64)
        n_pixels = flat_per_channel.shape[1]
        batch_mean = flat_per_channel.mean(axis=1)
        batch_var = flat_per_channel.var(axis=1)

        if self._count == 0:
            self._mean = batch_mean
            self._m2 = batch_var * n_pixels
            self._count = n_pixels
            return

        delta = batch_mean - self._mean
        total = self._count + n_pixels
        self._mean = self._mean + delta * (n_pixels / total)
        self._m2 = (
            self._m2
            + batch_var * n_pixels
            + delta**2 * self._count * n_pixels / total
        )
        self._count = total

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self._count == 0:
            raise ModelInputError("Cannot finalize normalization statistics: no train patches were seen")
        variance = self._m2 / self._count
        std = np.sqrt(variance)
        std = np.where(std < 1e-8, 1.0, std)
        means = self._mean.astype(np.float32).reshape(-1, 1, 1)
        stds = std.astype(np.float32).reshape(-1, 1, 1)
        return means, stds


def _cube_to_chw(cube: FeatureCube, feature_names: tuple[str, ...]) -> np.ndarray:
    """Stack a cube's per-band rasters into a (C, H, W) tensor. No flattening."""

    bands = [np.asarray(cube.features[name], dtype=np.float32) for name in feature_names]
    shapes = {band.shape for band in bands}
    if len(shapes) != 1:
        raise ModelInputError(
            f"All feature bands in a cube must share the same (H, W) shape, got {shapes}"
        )
    return np.stack(bands, axis=0)  # (C, H, W)


def _patch_grid(chw_shape: tuple[int, int, int], patch_size: int, stride: int):
    """Yield (row, col) top-left coordinates for every full-size patch in a (C, H, W) tensor.

    Only full-size patches are considered (no ragged edge patches), so every
    output patch has an identical shape. This iterates over the patch grid
    (e.g. ~43x43 = 1,849 steps for a 10980x10980 tile at 256/256), never over
    individual pixels.
    """

    _, height, width = chw_shape
    if height < patch_size or width < patch_size:
        return
    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            yield row, col


def _is_invalid_patch(patch: np.ndarray) -> bool:
    if patch.size == 0:
        return True
    if np.isnan(patch).any():
        return True
    if np.nanmax(patch) == np.nanmin(patch):
        return True
    return False


def _raw_patch_path(raw_dir: str, index: int) -> str:
    return os.path.join(raw_dir, f"raw_{index:06d}.npy")


def _final_patch_path(vision_dir: str, index: int) -> str:
    return os.path.join(vision_dir, f"patch_{index:06d}.npy")


def _splits(count: int) -> dict[str, list[int]]:
    if count <= 0:
        return {"train": [], "validation": [], "test": []}
    train_end = max(1, int(count * 0.7))
    validation_end = max(train_end, int(count * 0.85))
    indices = list(range(count))
    return {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }


def _json_safe(payload):
    """Recursively convert NumPy arrays/scalars to plain Python types for JSON export."""

    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, (np.floating, np.integer)):
        return payload.item()
    if isinstance(payload, dict):
        return {key: _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value) for value in payload]
    return payload