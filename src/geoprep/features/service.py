"""Stage 5: vegetation, SAR, and texture feature generation.

All feature computations are vectorized NumPy operations over full
raster bands. No function here iterates over individual pixels.
"""

from __future__ import annotations

import numpy as np

from geoprep.core.array_ops import ensure_same_shape, mean_filter, normalized_difference, safe_ratio
from geoprep.core.exceptions import FeatureGenerationError
from geoprep.core.logging import get_logger
from geoprep.core.models import AlignedObservation, Array2D, FeatureCube

_EPS = 1e-6

logger = get_logger("geoprep.features")


class FeatureGenerationService:
    """Generate deterministic geospatial feature cubes from aligned observations."""

    def __init__(self, logger_name: str = "geoprep.features") -> None:
        self.logger = get_logger(logger_name)

    def generate(self, observations: list[AlignedObservation]) -> list[FeatureCube]:
        """Generate all supported features for every observation."""

        return [self.generate_for_observation(observation) for observation in observations]

    def generate_for_observation(self, observation: AlignedObservation) -> FeatureCube:
        """Generate vegetation, SAR, and texture features for one observation."""

        self.logger.info(
            "Generating features",
            extra={"stage": "generate_for_observation"},
        )

        features: dict[str, Array2D] = {}

        if observation.optical is not None:
            red = _require_band(observation.optical.bands, ("red", "B04"))
            nir = _require_band(observation.optical.bands, ("nir", "B08"))
            blue = _require_band(observation.optical.bands, ("blue", "B02"), required=False)
            green = _require_band(observation.optical.bands, ("green", "B03"), required=False)
            swir = _require_band(observation.optical.bands, ("swir", "B11"), required=False)

            if blue is None:
                blue = red
            if green is None:
                green = red
            if swir is None:
                swir = red

            features["ndvi"] = normalized_difference(nir, red)
            features["evi"] = _evi(nir, red, blue)
            features["ndwi"] = normalized_difference(green, nir)
            features["savi"] = _savi(nir, red)
            features["gci"] = safe_ratio(nir, green)
            features["texture_contrast"] = _local_contrast(nir)
            features["texture_entropy"] = _entropy(nir)
            features["texture_homogeneity"] = _homogeneity(nir)

        if observation.sar is not None:
            vv = _require_band(observation.sar.bands, ("VV", "vv"), required=False)
            vh = _require_band(observation.sar.bands, ("VH", "vh"), required=False)

            if vv is not None:
                features["vv"] = vv
            if vh is not None:
                features["vh"] = vh
            if vv is not None and vh is not None:
                features["vh_vv_ratio"] = safe_ratio(vh, vv)

        if not features:
            raise FeatureGenerationError(f"No features generated for {observation.observation_id}")

        return FeatureCube(
            observation.observation_id,
            observation.acquisition_date,
            observation.crs,
            observation.resolution_m,
            features,
        )


def _require_band(bands: dict[str, Array2D], names: tuple[str, ...], required: bool = True) -> Array2D | None:
    """Look up a band by any of its accepted names.

    Uses `in` membership on the dict, never truthiness on the array itself,
    since `bool(ndarray)` raises ValueError for arrays with more than one element.
    """

    for name in names:
        if name in bands:
            return bands[name]
    if required:
        raise FeatureGenerationError(f"Missing required band. Expected one of: {names}")
    return None


def _evi(nir: Array2D, red: Array2D, blue: Array2D) -> Array2D:
    """Enhanced Vegetation Index: 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)."""

    ensure_same_shape(nir, red, blue)
    denominator = nir + 6.0 * red - 7.5 * blue + 1.0
    denominator = np.where(np.abs(denominator) < _EPS, _EPS, denominator)
    return 2.5 * (nir - red) / denominator


def _savi(nir: Array2D, red: Array2D, soil_factor: float = 0.5) -> Array2D:
    """Soil-Adjusted Vegetation Index: ((nir - red) * (1 + L)) / (nir + red + L)."""

    ensure_same_shape(nir, red)
    denominator = nir + red + soil_factor
    denominator = np.where(np.abs(denominator) < _EPS, _EPS, denominator)
    return ((nir - red) * (1.0 + soil_factor)) / denominator


def _local_contrast(array: Array2D) -> Array2D:
    """Local texture contrast: |value - local mean| via a vectorized mean filter."""

    smooth = mean_filter(array, radius=1)
    return np.abs(array - smooth)


def _entropy(array: Array2D) -> Array2D:
    """Per-pixel Shannon-style entropy term: -p * log(p), with values clamped to [0, 1]."""

    ensure_same_shape(array)
    probability = np.clip(array, 0.0, 1.0)
    safe_probability = np.maximum(probability, _EPS)
    return -probability * np.log(safe_probability)


def _homogeneity(array: Array2D) -> Array2D:
    """Local homogeneity: 1 / (1 + local contrast)."""

    contrast = _local_contrast(array)
    return 1.0 / (1.0 + contrast)
