"""GeoPrep end-to-end orchestration through AI-ready dataset export."""

from __future__ import annotations

from geoprep.alignment import AlignmentService
from geoprep.features import FeatureGenerationService
from geoprep.graph import SpatialGraphBuilder
from geoprep.model_inputs import ModelInputPreparer
from geoprep.preprocessing import PreprocessingService
from geoprep.temporal import TemporalSequenceBuilder
from geoprep.validation import DatasetValidator

from geoprep.core.models import (
    GraphDataset,
    RasterScene,
    TensorDataset,
)


class GeoPrepPipeline:
    """
    Runs GeoPrep preprocessing pipeline.

    Stages:
        3 - Preprocessing
        4 - Alignment
        5 - Feature Generation
        6 - Temporal Sequence Builder
        7 - Spatial Graph Builder (optional)
        8 - Validation
        9 - Model Input Export
    """

    def __init__(self) -> None:
        self.preprocessing = PreprocessingService()
        self.alignment = AlignmentService()
        self.features = FeatureGenerationService()
        self.temporal = TemporalSequenceBuilder()
        self.graph = SpatialGraphBuilder()
        self.validation = DatasetValidator()
        self.model_inputs = ModelInputPreparer()

    def run_from_scenes(
        self,
        optical: list[RasterScene],
        sar: list[RasterScene],
        graph_nodes: list[dict[str, object]],
    ) -> TensorDataset:

        if not optical and not sar:
            raise ValueError(
                "At least one optical or SAR scene is required."
            )

        reference_scene = (optical or sar)[0]

        target_crs = reference_scene.crs
        target_resolution = reference_scene.resolution_m
        target_shape = reference_scene.band_shape()

        # --------------------------------------------------
        # Stage 3
        # --------------------------------------------------

        clean_optical = [
            self.preprocessing.preprocess_optical(
                scene=scene,
                target_crs=target_crs,
                target_resolution_m=target_resolution,
                target_shape=target_shape,
            )
            for scene in optical
        ]

        clean_sar = [
            self.preprocessing.preprocess_sar(
                scene=scene,
                target_crs=target_crs,
                target_resolution_m=target_resolution,
                target_shape=target_shape,
            )
            for scene in sar
        ]

        # --------------------------------------------------
        # Stage 4
        # --------------------------------------------------

        aligned = self.alignment.align(
    optical_scenes=clean_optical,
    sar_scenes=clean_sar,
         )

        # --------------------------------------------------
        # Stage 5
        # --------------------------------------------------

        cubes = self.features.generate(aligned)

        # --------------------------------------------------
        # Stage 6
        # --------------------------------------------------

        sequence = self.temporal.build(cubes)

               # --------------------------------------------------
        # Stage 7 - Graph
        # --------------------------------------------------

        if graph_nodes:
            graph_dataset = self.graph.build(graph_nodes)
        else:
            # Empty graph for vision-only pipelines
            graph_dataset = GraphDataset(
                node_features=[],
                edge_index=[],
                edge_attr=[],
                graph_metadata={
                    "empty_graph": True
                },
            )

        # --------------------------------------------------
        # Stage 8 - Validation
        # --------------------------------------------------

        self.validation.validate(
            clean_optical + clean_sar,
            cubes,
            sequence,
            graph_dataset,
        )

        # --------------------------------------------------
        # Stage 9 - Model Inputs
        # --------------------------------------------------

        return self.model_inputs.prepare(
            cubes,
            sequence,
            graph_dataset,
        )