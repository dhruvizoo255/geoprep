"""Shared exception hierarchy for GeoPrep."""


class GeoPrepError(Exception):
    """Base exception for all GeoPrep failures."""


class DownloadError(GeoPrepError):
    """Raised when a download cannot be completed or verified."""


class ProcessingError(GeoPrepError):
    """Raised when raster preprocessing cannot be completed."""


class AlignmentError(GeoPrepError):
    """Raised when multimodal scenes cannot be aligned safely."""


class FeatureGenerationError(GeoPrepError):
    """Raised when feature generation inputs are incomplete."""


class TemporalSequenceError(GeoPrepError):
    """Raised when temporal sequence creation fails."""


class GraphConstructionError(GeoPrepError):
    """Raised when graph construction fails validation."""


class DatasetValidationError(GeoPrepError):
    """Raised when produced datasets are invalid."""


class ModelInputError(GeoPrepError):
    """Raised when AI-ready tensor export cannot be completed."""
