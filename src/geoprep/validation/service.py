"""Stage 8: dataset validation reports.

Cloud-fraction and array-presence checks are vectorized NumPy
operations. Graph edge-index checks operate on real (2, E) NumPy
arrays rather than nested Python lists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from geoprep.core.exceptions import DatasetValidationError
from geoprep.core.io import write_json
from geoprep.core.logging import get_logger
from geoprep.core.models import (
    FeatureCube,
    GraphDataset,
    RasterScene,
    SequenceDataset,
)


@dataclass(frozen=True)
class ValidationReport:
    """Validation result for a GeoPrep run."""

    passed: bool
    errors: list[str]
    warnings: list[str]
    metrics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DatasetValidator:
    """Validate GeoPrep outputs."""

    def __init__(
        self,
        max_cloud_fraction: float = 0.4,
        logger_name: str = "geoprep.validation",
    ) -> None:
        self.max_cloud_fraction = max_cloud_fraction
        self.logger = get_logger(logger_name)

    def validate(
        self,
        scenes: list[RasterScene],
        cubes: list[FeatureCube],
        sequence: SequenceDataset | None,
        graph: GraphDataset | None,
        report_path: str = "outputs/validation/validation_report.json",
    ) -> ValidationReport:

        self.logger.info(
            "Validating dataset",
            extra={"stage": "validate"},
        )

        errors: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, object] = {}

        self._validate_scenes(
            scenes,
            errors,
            warnings,
            metrics,
        )

        self._validate_cubes(
            cubes,
            errors,
            warnings,
            metrics,
        )

        if sequence is not None:
            self._validate_sequence(
                sequence,
                errors,
                warnings,
                metrics,
            )

        if graph is not None:
            self._validate_graph(
                graph,
                errors,
                warnings,
                metrics,
            )

        report = ValidationReport(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

        write_json(
            report_path,
            report.to_dict(),
        )

        if errors:
            raise DatasetValidationError("; ".join(errors))

        return report

    def _validate_scenes(
        self,
        scenes,
        errors,
        warnings,
        metrics,
    ):

        metrics["scene_count"] = len(scenes)

        ids = set()
        modalities = set()

        for scene in scenes:

            modalities.add(scene.modality)

            if scene.scene_id in ids:
                errors.append(
                    f"Duplicate scene: {scene.scene_id}"
                )

            ids.add(scene.scene_id)

            if not scene.crs:
                errors.append(
                    f"Missing CRS for {scene.scene_id}"
                )

            if not scene.bands:
                errors.append(
                    f"Missing bands for {scene.scene_id}"
                )

            if scene.cloud_mask is not None:

                fraction = float(
                    np.mean(scene.cloud_mask)
                )

                if fraction > self.max_cloud_fraction:
                    warnings.append(
                        f"High cloud cover for {scene.scene_id}: {fraction:.2f}"
                    )

        metrics["modalities"] = sorted(modalities)

        if "optical" not in modalities:
            warnings.append("Missing modality: optical")

        if "sar" not in modalities:
            warnings.append("Missing modality: sar")

    def _validate_cubes(
        self,
        cubes,
        errors,
        warnings,
        metrics,
    ):

        metrics["feature_cube_count"] = len(cubes)

        for cube in cubes:

            if not cube.features:

                errors.append(
                    f"Feature cube has no features: {cube.observation_id}"
                )

    def _validate_sequence(
        self,
        sequence,
        errors,
        warnings,
        metrics,
    ):

        metrics["sequence_count"] = len(sequence.sequences)

        for i, timestamps in enumerate(sequence.timestamps):

            if timestamps != sorted(timestamps):
                errors.append(
                    f"Sequence {i} timestamps are not chronological"
                )

            if len(timestamps) != len(set(timestamps)):
                warnings.append(
                    f"Sequence {i} contains duplicate timestamps"
                )

    def _validate_graph(
        self,
        graph,
        errors,
        warnings,
        metrics,
    ):
        """
        Validate graph.

        Empty graphs are VALID for vision-only pipelines.
        """

        if graph.graph_metadata.get("empty_graph", False):

            metrics["graph_node_count"] = 0
            metrics["graph_edge_count"] = 0

            warnings.append(
                "Vision-only pipeline: graph validation skipped."
            )

            return

        node_count = len(graph.node_features)

        edge_index = np.asarray(graph.edge_index)
        edge_attr = np.asarray(graph.edge_attr)

        metrics["graph_node_count"] = node_count
        metrics["graph_edge_count"] = (
            int(edge_attr.shape[0])
            if edge_attr.ndim > 0
            else 0
        )

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            errors.append(
                "edge_index must have shape [2, num_edges]"
            )
            return

        num_edges = edge_index.shape[1]

        if num_edges != edge_attr.shape[0]:
            errors.append(
                "edge_index and edge_attr lengths do not match"
            )

        if num_edges == 0:
            return

        all_nodes = np.concatenate(
            [edge_index[0], edge_index[1]]
        )

        invalid = all_nodes[
            (all_nodes < 0)
            | (all_nodes >= node_count)
        ]

        if invalid.size:

            for node in np.unique(invalid):

                errors.append(
                    f"Graph edge references invalid node index: {int(node)}"
                )