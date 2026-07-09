"""Configuration models and YAML loading for GeoPrep discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from discovery.exceptions import ConfigurationError


@dataclass(frozen=True)
class DiscoveryRequest:
    """User-scoped discovery request."""

    aoi_name: str
    start_date: date
    end_date: date
    aoi_geojson: dict[str, Any] | None = None
    season: str | None = None
    modalities: tuple[str, ...] = ("optical", "sar", "weather", "ancillary")

    def __post_init__(self) -> None:
        if self.end_date < self.start_date:
            raise ConfigurationError("end_date must be on or after start_date")
        if not self.aoi_name and not self.aoi_geojson:
            raise ConfigurationError("Either aoi_name or aoi_geojson must be provided")


@dataclass(frozen=True)
class ProviderConfig:
    """Runtime settings for one discovery provider."""

    name: str
    enabled: bool = True
    kind: str = "static"
    endpoint: str | None = None
    collections: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()
    requires_credentials: bool = False
    credential_env: str | None = None
    max_items: int = 100


@dataclass(frozen=True)
class DiscoveryConfig:
    """Complete configuration for Stage 1 discovery."""

    request: DiscoveryRequest
    providers: tuple[ProviderConfig, ...]
    defaults: dict[str, Any] = field(default_factory=dict)
    output_metadata_path: str = "work/discovery/discovery_report.json"
    log_file: str = "work/logs/discovery.log"
    log_level: str = "INFO"
    required_modalities: tuple[str, ...] = ("optical", "sar", "weather", "ancillary")


def load_config(path: str | Path) -> DiscoveryConfig:
    """Load discovery configuration from a simple YAML file."""

    raw = _load_simple_yaml(Path(path))
    try:
        request_data = raw["request"]
        defaults = raw.get("defaults", {})
        logging_config = raw.get("logging", {})
        validation_config = raw.get("validation", {})
        provider_items = raw["providers"]
    except KeyError as exc:
        raise ConfigurationError(f"Missing required config key: {exc.args[0]}") from exc

    request = DiscoveryRequest(
        aoi_name=str(request_data.get("aoi_name", "")),
        aoi_geojson=request_data.get("aoi_geojson"),
        season=request_data.get("season"),
        start_date=_parse_date(str(request_data["start_date"])),
        end_date=_parse_date(str(request_data["end_date"])),
        modalities=tuple(request_data.get("modalities", [])) or ("optical", "sar", "weather", "ancillary"),
    )
    providers = tuple(_provider_from_dict(item) for item in provider_items)
    return DiscoveryConfig(
        request=request,
        providers=providers,
        defaults=defaults,
        output_metadata_path=str(raw.get("output_metadata_path", "work/discovery/discovery_report.json")),
        log_file=str(logging_config.get("file", "work/logs/discovery.log")),
        log_level=str(logging_config.get("level", "INFO")),
        required_modalities=tuple(validation_config.get("required_modalities", request.modalities)),
    )


def _provider_from_dict(data: dict[str, Any]) -> ProviderConfig:
    return ProviderConfig(
        name=str(data["name"]),
        enabled=bool(data.get("enabled", True)),
        kind=str(data.get("kind", "static")),
        endpoint=data.get("endpoint"),
        collections=tuple(data.get("collections", [])),
        modalities=tuple(data.get("modalities", [])),
        requires_credentials=bool(data.get("requires_credentials", False)),
        credential_env=data.get("credential_env"),
        max_items=int(data.get("max_items", 100)),
    )


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid ISO date: {value}") from exc


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the subset of YAML used by configs/discovery.yaml.

    PyYAML is intentionally not required, keeping Stage 1 runnable in the
    bundled Codex Python runtime.
    """

    if not path.exists():
        raise ConfigurationError(f"Config file does not exist: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for raw_line in lines:
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if text.startswith("- "):
            if not isinstance(parent, list):
                raise ConfigurationError(f"List item without list parent: {raw_line}")
            item_text = text[2:].strip()
            if ":" in item_text:
                key, value = _split_key_value(item_text)
                item: dict[str, Any] = {key: _parse_scalar(value)}
                parent.append(item)
                stack.append((indent, item))
            else:
                parent.append(_parse_scalar(item_text))
            continue

        key, value = _split_key_value(text)
        if value == "":
            container: Any = [] if _next_content_is_list(lines, raw_line) else {}
            if isinstance(parent, dict):
                parent[key] = container
            else:
                raise ConfigurationError(f"Mapping key under non-mapping parent: {raw_line}")
            stack.append((indent, container))
        else:
            if not isinstance(parent, dict):
                raise ConfigurationError(f"Scalar key under non-mapping parent: {raw_line}")
            parent[key] = _parse_scalar(value)

    return root


def _next_content_is_list(lines: list[str], current_line: str) -> bool:
    current_index = lines.index(current_line)
    for next_line in lines[current_index + 1 :]:
        stripped = next_line.split("#", 1)[0].strip()
        if stripped:
            return stripped.startswith("- ")
    return False


def _split_key_value(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ConfigurationError(f"Expected key/value pair: {text}")
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _parse_scalar(value: str) -> Any:
    if value == "":
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value
