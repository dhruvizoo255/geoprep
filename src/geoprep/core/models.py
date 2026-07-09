"""
Typed data models shared across GeoPrep stages.

Production version supporting NumPy arrays.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------

Array2D = np.ndarray


# -----------------------------------------------------------------------------
# Download Stage
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DownloadArtifact:
    """Represents one downloaded satellite artifact."""

    product_id: str
    modality: str
    source_url: str | None
    local_path: str
    checksum_sha256: str
    bytes_downloaded: int
    resumed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Raster Scene
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RasterScene:
    """
    Analysis-ready raster scene.

    Stores multiple raster bands as NumPy arrays.
    """

    scene_id: str
    modality: str
    acquisition_date: str
    crs: str
    resolution_m: float

    bands: dict[str, Array2D]

    cloud_mask: np.ndarray | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        if not self.bands:
            return 0

        first_band = next(iter(self.bands.values()))
        return first_band.shape[0]

    @property
    def width(self) -> int:
        if not self.bands:
            return 0

        first_band = next(iter(self.bands.values()))
        return first_band.shape[1]

    def band_shape(self) -> tuple[int, int]:
        """
        Returns

        (height, width)
        """

        return (self.height, self.width)

    @property
    def band_names(self) -> list[str]:
        return list(self.bands.keys())

    @property
    def num_bands(self) -> int:
        return len(self.bands)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "modality": self.modality,
            "acquisition_date": self.acquisition_date,
            "crs": self.crs,
            "resolution_m": self.resolution_m,
            "bands": self.band_names,
            "height": self.height,
            "width": self.width,
            "metadata": self.metadata,
        }


# -----------------------------------------------------------------------------
# Alignment
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AlignedObservation:
    """
    Optical + SAR observation after alignment.
    """

    observation_id: str
    acquisition_date: str

    optical: RasterScene | None
    sar: RasterScene | None

    crs: str
    resolution_m: float

    metadata: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureCube:
    """
    Generated feature cube.
    """

    observation_id: str
    acquisition_date: str

    crs: str
    resolution_m: float

    features: dict[str, Array2D]

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "feature_names": list(self.features.keys()),
            "metadata": self.metadata,
        }


# -----------------------------------------------------------------------------
# Temporal Dataset
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SequenceDataset:
    """
    Temporal Transformer dataset.
    """

    sequence_id: str

    feature_names: tuple[str, ...]

    sequences: np.ndarray

    masks: np.ndarray

    timestamps: list[list[str]]

    metadata: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Graph Dataset
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphDataset:
    """
    Graph Neural Network dataset.
    """

    node_features: np.ndarray

    edge_index: np.ndarray

    edge_attr: np.ndarray

    graph_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "graph_metadata": self.graph_metadata,
        }


# -----------------------------------------------------------------------------
# Final Dataset
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorDataset:
    """
    Final exported dataset.
    """

    vision_transformer: dict[str, Any]

    temporal_transformer: dict[str, Any]

    graph_neural_network: dict[str, Any]

    splits: dict[str, list[int]]

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "vision_transformer": self.vision_transformer,
            "temporal_transformer": self.temporal_transformer,
            "graph_neural_network": self.graph_neural_network,
            "splits": self.splits,
            "metadata": self.metadata,
        }