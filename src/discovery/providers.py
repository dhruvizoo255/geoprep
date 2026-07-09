"""Discovery provider implementations for Stage 1."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Protocol

from discovery.config import DiscoveryRequest, ProviderConfig
from discovery.exceptions import ProviderError


@dataclass(frozen=True)
class DatasetCandidate:
    """Validated raw-dataset metadata discovered during Stage 1."""

    provider: str
    source: str
    collection: str
    product_id: str
    modality: str
    acquisition_date: str | None
    start_date: str | None
    end_date: str | None
    spatial_resolution_m: float | None
    crs: str | None
    bands: tuple[str, ...] = ()
    cloud_cover_percent: float | None = None
    product_level: str | None = None
    preprocessing_state: str = "raw_or_provider_supplied"
    access_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bands"] = list(self.bands)
        return data


@dataclass(frozen=True)
class ProviderResult:
    """Result from one provider, including fallbacks or warnings."""

    provider: str
    candidates: tuple[DatasetCandidate, ...]
    warnings: tuple[str, ...] = ()
    fallbacks: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "warnings": list(self.warnings),
            "fallbacks": list(self.fallbacks),
        }


class DiscoveryProvider(Protocol):
    """Protocol implemented by all Stage 1 providers."""

    name: str

    def discover(self, request: DiscoveryRequest) -> ProviderResult:
        """Discover raw dataset metadata for the request."""


class HttpTransport:
    """Small JSON HTTP client used by STAC providers."""

    def post_json(self, url: str, payload: dict[str, Any], timeout_seconds: int = 30) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ProviderError(f"HTTP discovery failed for {url}: {exc}") from exc


class StacProvider:
    """Provider that discovers datasets from a STAC Item Search endpoint."""

    def __init__(self, config: ProviderConfig, transport: HttpTransport | None = None) -> None:
        if not config.endpoint:
            raise ProviderError(f"STAC provider {config.name} requires an endpoint")
        self.config = config
        self.name = config.name
        self.transport = transport or HttpTransport()

    def discover(self, request: DiscoveryRequest) -> ProviderResult:
        warnings: list[str] = []
        if self.config.requires_credentials and not _has_credential(self.config):
            return ProviderResult(
                provider=self.name,
                candidates=(),
                warnings=(f"{self.name} skipped because credential {self.config.credential_env} is not set",),
            )

        payload = {
            "collections": list(self.config.collections),
            "datetime": f"{request.start_date.isoformat()}/{request.end_date.isoformat()}",
            "limit": self.config.max_items,
        }
        if request.aoi_geojson:
            payload["intersects"] = request.aoi_geojson

        data = self.transport.post_json(self.config.endpoint, payload)
        features = data.get("features", [])
        if not isinstance(features, list):
            raise ProviderError(f"{self.name} returned invalid STAC response: features must be a list")

        candidates = tuple(self._candidate_from_feature(feature) for feature in features)
        if not candidates:
            warnings.append(f"{self.name} returned no items for requested AOI/date range")
        return ProviderResult(provider=self.name, candidates=candidates, warnings=tuple(warnings))

    def _candidate_from_feature(self, feature: dict[str, Any]) -> DatasetCandidate:
        properties = feature.get("properties", {})
        collection = str(feature.get("collection") or properties.get("collection") or "unknown")
        modality = _infer_modality(collection, self.config.modalities)
        assets = feature.get("assets", {})
        access_url = _first_asset_href(assets)
        resolution = _first_number(
            properties.get("gsd"),
            properties.get("proj:epsg"),
            None,
        )
        crs = _infer_crs(properties)
        bands = tuple(_infer_bands(properties, assets))
        return DatasetCandidate(
            provider=self.name,
            source=self.name,
            collection=collection,
            product_id=str(feature.get("id") or properties.get("product_id") or "unknown"),
            modality=modality,
            acquisition_date=_date_part(properties.get("datetime")),
            start_date=_date_part(properties.get("start_datetime")),
            end_date=_date_part(properties.get("end_datetime")),
            spatial_resolution_m=resolution if properties.get("gsd") is not None else None,
            crs=crs,
            bands=bands,
            cloud_cover_percent=_first_number(
                properties.get("eo:cloud_cover"),
                properties.get("cloud_cover"),
                properties.get("landsat:cloud_cover_land"),
            ),
            product_level=_infer_product_level(collection, properties),
            access_url=access_url,
            metadata={
                "stac_version": feature.get("stac_version"),
                "platform": properties.get("platform"),
                "instrument": properties.get("instruments"),
            },
        )


class StaticCatalogProvider:
    """Provider for portals that require manual/API credentials or later integration."""

    _CATALOG: dict[str, dict[str, Any]] = {
        "resourcesat": {
            "source": "Bhuvan/Bhoonidhi",
            "collection": "Resourcesat LISS/AWiFS",
            "modality": "optical",
            "bands": ("green", "red", "nir", "swir"),
            "spatial_resolution_m": 23.5,
            "crs": None,
        },
        "awifs": {
            "source": "Bhuvan/Bhoonidhi",
            "collection": "AWiFS",
            "modality": "optical",
            "bands": ("green", "red", "nir", "swir"),
            "spatial_resolution_m": 56.0,
            "crs": None,
        },
        "eos-04": {
            "source": "Bhoonidhi",
            "collection": "EOS-04 SAR",
            "modality": "sar",
            "bands": ("HH", "HV", "VV", "VH"),
            "spatial_resolution_m": None,
            "crs": None,
        },
        "modis": {
            "source": "NASA/LP DAAC",
            "collection": "MODIS surface reflectance and vegetation products",
            "modality": "optical",
            "bands": ("red", "nir", "blue", "swir", "ndvi", "evi"),
            "spatial_resolution_m": 250.0,
            "crs": "Sinusoidal",
        },
        "weather": {
            "source": "Weather data catalog",
            "collection": "Rainfall/Temperature/Humidity/Solar Radiation/ET0",
            "modality": "weather",
            "bands": ("rainfall", "temperature", "humidity", "solar_radiation", "et0"),
            "spatial_resolution_m": None,
            "crs": None,
        },
        "ancillary-gis": {
            "source": "Ancillary GIS catalog",
            "collection": "Administrative boundaries/soil/irrigation layers",
            "modality": "ancillary",
            "bands": (),
            "spatial_resolution_m": None,
            "crs": None,
        },
    }

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.name = config.name

    def discover(self, request: DiscoveryRequest) -> ProviderResult:
        if self.config.requires_credentials and not _has_credential(self.config):
            return ProviderResult(
                provider=self.name,
                candidates=(),
                warnings=(f"{self.name} skipped because credential {self.config.credential_env} is not set",),
                fallbacks=(f"Use portal/API credentials for {self.name} or configure an open STAC endpoint.",),
            )

        catalog_key = self.name.lower()
        template = self._CATALOG.get(catalog_key)
        if not template:
            raise ProviderError(f"No static catalog template registered for provider {self.name}")

        candidate = DatasetCandidate(
            provider=self.name,
            source=str(template["source"]),
            collection=str(template["collection"]),
            product_id=f"{self.name}:{request.aoi_name}:{request.start_date.isoformat()}:{request.end_date.isoformat()}",
            modality=str(template["modality"]),
            acquisition_date=None,
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat(),
            spatial_resolution_m=template["spatial_resolution_m"],
            crs=template["crs"],
            bands=tuple(template["bands"]),
            product_level="catalog_reference",
            access_url=None,
            metadata={
                "aoi_name": request.aoi_name,
                "season": request.season,
                "note": "Catalog-level discovery record; exact scenes require provider API credentials or portal search.",
            },
        )
        return ProviderResult(provider=self.name, candidates=(candidate,))


def build_provider(config: ProviderConfig, transport: HttpTransport | None = None) -> DiscoveryProvider:
    """Build a concrete provider from configuration."""

    if config.kind == "stac":
        return StacProvider(config, transport=transport)
    if config.kind == "static":
        return StaticCatalogProvider(config)
    raise ProviderError(f"Unsupported provider kind for {config.name}: {config.kind}")


def _has_credential(config: ProviderConfig) -> bool:
    return bool(config.credential_env and os.getenv(config.credential_env))


def _infer_modality(collection: str, configured: tuple[str, ...]) -> str:
    if configured:
        return configured[0]
    lower = collection.lower()
    if "sentinel-1" in lower or "sar" in lower:
        return "sar"
    if "weather" in lower:
        return "weather"
    if "boundary" in lower or "soil" in lower:
        return "ancillary"
    return "optical"


def _infer_crs(properties: dict[str, Any]) -> str | None:
    epsg = properties.get("proj:epsg")
    if epsg is not None:
        return f"EPSG:{epsg}"
    return properties.get("proj:wkt2") or properties.get("crs")


def _infer_product_level(collection: str, properties: dict[str, Any]) -> str | None:
    for key in ("processing:level", "sentinel:processing_baseline", "landsat:correction"):
        if properties.get(key):
            return str(properties[key])
    lower = collection.lower()
    if "l2a" in lower or "surface" in lower:
        return "surface_reflectance"
    if "l1c" in lower:
        return "top_of_atmosphere"
    return None


def _infer_bands(properties: dict[str, Any], assets: dict[str, Any]) -> list[str]:
    eo_bands = properties.get("eo:bands")
    if isinstance(eo_bands, list):
        names = [str(item.get("name")) for item in eo_bands if isinstance(item, dict) and item.get("name")]
        if names:
            return names
    return sorted(str(name) for name in assets.keys())


def _first_asset_href(assets: dict[str, Any]) -> str | None:
    for asset in assets.values():
        if isinstance(asset, dict) and asset.get("href"):
            return str(asset["href"])
    return None


def _date_part(value: Any) -> str | None:
    if not value:
        return None
    return str(value)[:10]


def _first_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None
