"""Stage 7: PyTorch Geometric-compatible graph construction.

Edge and node tensors are built with vectorized NumPy operations
(pairwise broadcasting) instead of an O(N^2) pure-Python double loop.
"""

from __future__ import annotations

import numpy as np

from geoprep.core.exceptions import GraphConstructionError
from geoprep.core.logging import get_logger
from geoprep.core.models import GraphDataset

_EDGE_ATTRIBUTE_NAMES = ("distance_m", "shared_irrigation", "soil_similarity", "rainfall_similarity")


class SpatialGraphBuilder:
    """Build graph tensors with proximity and similarity edge attributes."""

    def __init__(self, proximity_threshold_m: float = 5000.0, logger_name: str = "geoprep.graph") -> None:
        self.proximity_threshold_m = proximity_threshold_m
        self.logger = get_logger(logger_name)

    def build(
        self,
        nodes: list[dict[str, object]],
        k_neighbors: int | None = None,
    ) -> GraphDataset:
        """Build edge_index, edge_attr, node_features, and graph metadata.

        By default, an edge is created between any pair of distinct nodes where
        at least one of the following holds: nodes are within
        `proximity_threshold_m`, they share an irrigation command area, or their
        soil/rainfall similarity exceeds 0.8 (the original "distance graph"
        behaviour, preserved exactly).

        If `k_neighbors` is provided, the graph is additionally restricted to a
        k-nearest-neighbour graph: for each node, only edges to its
        `k_neighbors` spatially nearest candidates (that also satisfy the
        condition above) are kept. This bounds edge count to O(N * k) for large
        node counts, at the cost of asymmetry (i in neighbours of j does not
        guarantee j in neighbours of i) which is standard for k-NN graphs.
        """

        if not nodes:
            raise GraphConstructionError("At least one node is required")

        self.logger.info(
            "Building spatial graph",
            extra={"stage": "build"},
        )

        node_count = len(nodes)
        node_features = np.array(
            [list(map(float, node.get("features", []))) for node in nodes],
            dtype=np.float32,
        )

        x = np.array([float(node.get("x", 0.0)) for node in nodes], dtype=np.float64)
        y = np.array([float(node.get("y", 0.0)) for node in nodes], dtype=np.float64)
        rainfall = np.array([float(node.get("rainfall_mm", 0.0)) for node in nodes], dtype=np.float64)
        irrigation_ids = np.array([node.get("irrigation_id") for node in nodes], dtype=object)
        soil_types = np.array([node.get("soil_type") for node in nodes], dtype=object)

        distance = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])

        irrigation_present = np.array([bool(v) for v in irrigation_ids], dtype=bool)
        shared_irrigation = (
            (irrigation_ids[:, None] == irrigation_ids[None, :])
            & irrigation_present[:, None]
            & irrigation_present[None, :]
        ).astype(np.float64)

        soil_present = np.array([bool(v) for v in soil_types], dtype=bool)
        same_soil = (soil_types[:, None] == soil_types[None, :]) & soil_present[:, None] & soil_present[None, :]
        feature_cosine = _pairwise_cosine_similarity(node_features)
        soil_similarity = np.where(same_soil, 1.0, np.maximum(0.0, feature_cosine))

        rainfall_similarity = 1.0 - np.minimum(1.0, np.abs(rainfall[:, None] - rainfall[None, :]) / 1000.0)

        candidate_mask = (
            (distance <= self.proximity_threshold_m)
            | (shared_irrigation > 0)
            | (soil_similarity > 0.8)
            | (rainfall_similarity > 0.8)
        )
        np.fill_diagonal(candidate_mask, False)

        if k_neighbors is not None:
            candidate_mask = _restrict_to_k_nearest(candidate_mask, distance, k_neighbors)

        sources, targets = np.nonzero(candidate_mask)
        edge_index = np.stack([sources, targets], axis=0).astype(np.int64)
        edge_attr = np.stack(
            [
                distance[sources, targets],
                shared_irrigation[sources, targets],
                soil_similarity[sources, targets],
                rainfall_similarity[sources, targets],
            ],
            axis=1,
        ).astype(np.float32)

        return GraphDataset(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            graph_metadata={
                "node_count": node_count,
                "edge_count": int(edge_attr.shape[0]),
                "edge_attributes": list(_EDGE_ATTRIBUTE_NAMES),
                "format": "pytorch_geometric",
                "k_neighbors": k_neighbors,
                "proximity_threshold_m": self.proximity_threshold_m,
            },
        )


def _pairwise_cosine_similarity(features: np.ndarray) -> np.ndarray:
    """Full pairwise cosine similarity matrix via a single vectorized matmul.

    Rows with (near-)zero norm produce zero similarity against every other
    row, matching the original per-pair guard against degenerate vectors.
    """

    if features.size == 0 or features.shape[1] == 0:
        n = features.shape[0]
        return np.zeros((n, n), dtype=np.float64)

    norms = np.linalg.norm(features, axis=1)
    safe_norms = np.where(norms < 1e-8, 1.0, norms)
    normalized = features / safe_norms[:, None]
    similarity = normalized @ normalized.T
    degenerate = norms < 1e-8
    similarity[degenerate, :] = 0.0
    similarity[:, degenerate] = 0.0
    return similarity


def _restrict_to_k_nearest(candidate_mask: np.ndarray, distance: np.ndarray, k_neighbors: int) -> np.ndarray:
    """Keep, per row, only the k nearest True entries in `candidate_mask` (by `distance`)."""

    if k_neighbors <= 0:
        raise GraphConstructionError("k_neighbors must be a positive integer")

    restricted = np.zeros_like(candidate_mask)
    masked_distance = np.where(candidate_mask, distance, np.inf)
    k = min(k_neighbors, candidate_mask.shape[1])
    nearest_indices = np.argsort(masked_distance, axis=1)[:, :k]
    rows = np.repeat(np.arange(candidate_mask.shape[0]), k)
    cols = nearest_indices.ravel()
    valid = np.isfinite(masked_distance[rows, cols])
    restricted[rows[valid], cols[valid]] = True
    return restricted
