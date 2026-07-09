"""Stage 6: transformer-ready temporal sequences.

Builds `SequenceDataset.sequences` / `.masks` as real NumPy arrays
(matching the dataclass's own type hints), computed via vectorized
NumPy reductions rather than per-pixel Python loops.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from geoprep.core.array_ops import flatten
from geoprep.core.exceptions import TemporalSequenceError
from geoprep.core.logging import get_logger
from geoprep.core.models import FeatureCube, SequenceDataset


class TemporalSequenceBuilder:
    """Build chronological fixed-length temporal sequences with masks."""

    def __init__(self, sequence_length: int = 8, interval_days: int = 10, logger_name: str = "geoprep.temporal") -> None:
        if sequence_length <= 0:
            raise TemporalSequenceError("sequence_length must be positive")
        self.sequence_length = sequence_length
        self.interval_days = interval_days
        self.logger = get_logger(logger_name)

    def build(self, cubes: list[FeatureCube]) -> SequenceDataset:
        """Sort, gap-fill, pad, and mask feature cubes into a fixed-length sequence."""

        if not cubes:
            raise TemporalSequenceError("At least one feature cube is required")

        self.logger.info(
            "Building temporal sequence",
            extra={"stage": "build"},
        )

        ordered = sorted(cubes, key=lambda cube: cube.acquisition_date)
        feature_names = tuple(sorted(ordered[0].features.keys()))

        vectors_by_date: dict[str, np.ndarray] = {
            cube.acquisition_date: _vectorize(cube, feature_names) for cube in ordered
        }
        width = len(feature_names)

        timeline = self._timeline(
            date.fromisoformat(ordered[0].acquisition_date),
            date.fromisoformat(ordered[-1].acquisition_date),
        )

        vectors: list[np.ndarray] = []
        mask: list[int] = []
        for timestamp in timeline:
            key = timestamp.isoformat()
            if key in vectors_by_date:
                vectors.append(vectors_by_date[key])
                mask.append(1)
            else:
                vectors.append(_interpolate_vector(vectors_by_date, key, width))
                mask.append(0)

        while len(vectors) < self.sequence_length:
            vectors.append(np.zeros(width, dtype=np.float32))
            mask.append(0)
            timeline.append(timeline[-1] + timedelta(days=self.interval_days))

        sequence_array = np.stack(vectors[: self.sequence_length], axis=0).astype(np.float32)
        mask_array = np.array(mask[: self.sequence_length], dtype=np.int64)

        return SequenceDataset(
            sequence_id="temporal_sequence_0",
            feature_names=feature_names,
            sequences=sequence_array[None, ...],
            masks=mask_array[None, ...],
            timestamps=[[item.isoformat() for item in timeline[: self.sequence_length]]],
            metadata={"interval_days": self.interval_days, "sequence_length": self.sequence_length},
        )

    def _timeline(self, start: date, end: date) -> list[date]:
        values: list[date] = []
        current = start
        while current <= end and len(values) < self.sequence_length:
            values.append(current)
            current += timedelta(days=self.interval_days)
        return values or [start]


def _vectorize(cube: FeatureCube, feature_names: tuple[str, ...]) -> np.ndarray:
    """Per-feature scalar summary (mean over the raster) for one timestep, vectorized via np.mean."""

    return np.array(
        [float(np.mean(flatten(cube.features[name]))) for name in feature_names],
        dtype=np.float32,
    )


def _interpolate_vector(vectors_by_date: dict[str, np.ndarray], missing_date: str, width: int) -> np.ndarray:
    """Linearly interpolate a missing timestep's feature vector from its nearest known neighbours."""

    before = [key for key in vectors_by_date if key < missing_date]
    after = [key for key in vectors_by_date if key > missing_date]

    if before and after:
        left = vectors_by_date[max(before)]
        right = vectors_by_date[min(after)]
        return ((left + right) / 2.0).astype(np.float32)
    if before:
        return vectors_by_date[max(before)]
    if after:
        return vectors_by_date[min(after)]
    return np.zeros(width, dtype=np.float32)
