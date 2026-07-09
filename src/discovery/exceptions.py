"""Custom exceptions for Stage 1 discovery."""


class DiscoveryError(Exception):
    """Base exception for discovery failures."""


class ConfigurationError(DiscoveryError):
    """Raised when discovery configuration is invalid."""


class ProviderError(DiscoveryError):
    """Raised when a provider cannot complete discovery."""


class ValidationError(DiscoveryError):
    """Raised when discovered metadata fails validation."""
