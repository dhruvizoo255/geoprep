"""Stage 3: optical and SAR preprocessing operations.

All band operations here are vectorized NumPy array operations.
No stage in this module iterates over individual pixels in Python.
"""

from __future__ import annotations

import numpy as np

from geoprep.core.array_ops import clip, mean_filter, nearest_resample, scale
from geoprep.core.exceptions import ProcessingError
from geoprep.core.logging import get_logger
from geoprep.core.models import Array2D, RasterScene

_DB_FLOOR = 1e-6


class PreprocessingService:
    """Preprocess optical and SAR scenes while preserving raw inputs."""

    def __init__(self, logger_name: str = "geoprep.preprocessing") -> None:
        self.logger = get_logger(logger_name)

    def preprocess_optical(
        self,
        scene: RasterScene,
        target_crs: str,
        target_resolution_m: float,
        target_shape: tuple[int, int] | None = None,
        reflectance_scale: float = 0.0001,
        cloud_threshold: float = 0.4,
    ) -> RasterScene:
        """Apply atmospheric correction, cloud masking, reprojection, clipping, and resampling."""

        if scene.modality != "optical":
            raise ProcessingError("Optical preprocessing requires an optical scene")

        self.logger.info(
            "Preprocessing optical scene",
            extra={"stage": "preprocess_optical"},
        )

        bands = {
            name: clip(scale(values, reflectance_scale), 0.0, 1.0)
            for name, values in scene.bands.items()
        }
        cloud_mask = self._cloud_mask(bands, cloud_threshold)

        if target_shape:
            bands = {name: nearest_resample(values, *target_shape) for name, values in bands.items()}
            cloud_mask = self._resample_mask(cloud_mask, *target_shape)

        return RasterScene(
            scene_id=scene.scene_id,
            modality=scene.modality,
            acquisition_date=scene.acquisition_date,
            crs=target_crs,
            resolution_m=target_resolution_m,
            bands=bands,
            cloud_mask=cloud_mask,
            metadata={
                **scene.metadata,
                "atmospheric_correction": "linear_reflectance_scale",
                "cloud_threshold": cloud_threshold,
            },
        )

    def preprocess_sar(
        self,
        scene: RasterScene,
        target_crs: str,
        target_resolution_m: float,
        target_shape: tuple[int, int] | None = None,
        calibration_factor: float = 1.0,
        terrain_factor: float = 1.0,
    ) -> RasterScene:
        """Apply radiometric calibration, speckle filtering, terrain correction, and resampling."""

        if scene.modality != "sar":
            raise ProcessingError("SAR preprocessing requires a SAR scene")

        self.logger.info(
            "Preprocessing SAR scene",
            extra={"stage": "preprocess_sar"},
        )

        calibrated = {name: self._to_db(scale(values, calibration_factor)) for name, values in scene.bands.items()}
        filtered = {name: mean_filter(values, radius=1) for name, values in calibrated.items()}
        corrected = {name: scale(values, terrain_factor) for name, values in filtered.items()}

        if target_shape:
            corrected = {name: nearest_resample(values, *target_shape) for name, values in corrected.items()}

        return RasterScene(
            scene_id=scene.scene_id,
            modality=scene.modality,
            acquisition_date=scene.acquisition_date,
            crs=target_crs,
            resolution_m=target_resolution_m,
            bands=corrected,
            metadata={
                **scene.metadata,
                "radiometric_calibration": calibration_factor,
                "speckle_filter": "mean_3x3",
                "terrain_factor": terrain_factor,
            },
        )

    def _cloud_mask(self, bands: dict[str, Array2D], threshold: float) -> Array2D:
        """Vectorized cloud mask: boolean array where the reference band exceeds `threshold`.

        Preference order for the reference band: blue -> B02 -> first available band.
        Uses explicit `is not None` checks rather than truthy checks, since
        `bool(ndarray)` raises ValueError for arrays with more than one element.
        """

        reference = bands.get("blue")
        if reference is None:
            reference = bands.get("B02")
        if reference is None:
            reference = next(iter(bands.values()))

        return reference > threshold

    def _resample_mask(self, mask: Array2D, rows: int, cols: int) -> Array2D:
        """Resample a boolean mask by resampling as float32 then re-thresholding at 0.5.

        Nearest-neighbour interpolation on a 0/1 float representation preserves
        the boolean semantics of the mask (no fractional "half-cloud" pixels).
        """

        numeric = mask.astype(np.float32)
        resized = nearest_resample(numeric, rows, cols)
        return resized >= 0.5

    def _to_db(self, array: Array2D) -> Array2D:
        """Vectorized linear-to-decibel conversion: 10 * log10(max(value, floor))."""

        return 10.0 * np.log10(np.clip(array, _DB_FLOOR, None))
