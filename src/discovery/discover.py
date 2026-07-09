"""Stage 1 orchestration: discover and validate raw dataset metadata."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from discovery.config import DiscoveryConfig, load_config
from discovery.exceptions import ProviderError, ValidationError
from discovery.logger import configure_logger
from discovery.providers import DatasetCandidate, ProviderResult, build_provider


@dataclass(frozen=True)
class DiscoveryReport:
    """Complete Stage 1 discovery output."""

    generated_at: str
    aoi_name: str
    start_date: str
    end_date: str
    defaults_used: dict[str, object]
    candidates: tuple[DatasetCandidate, ...]
    warnings: tuple[str, ...] = ()
    fallbacks: tuple[str, ...] = ()
    provider_results: tuple[ProviderResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": "stage_1_satellite_data_discovery",
            "generated_at": self.generated_at,
            "request": {
                "aoi_name": self.aoi_name,
                "start_date": self.start_date,
                "end_date": self.end_date,
            },
            "defaults_used": self.defaults_used,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "warnings": list(self.warnings),
            "fallbacks": list(self.fallbacks),
            "provider_results": [result.to_dict() for result in self.provider_results],
        }

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class DiscoveryService:
    """Execute Stage 1 only: discover raw data and validate metadata."""

    def __init__(self, config: DiscoveryConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or configure_logger(
            level=config.log_level,
            log_file=config.log_file,
        )

    def run(self) -> DiscoveryReport:
        self.logger.info("Starting Stage 1 discovery for AOI=%s", self.config.request.aoi_name)
        self.logger.info("Using defaults: %s", self.config.defaults)

        provider_results: list[ProviderResult] = []
        warnings: list[str] = []
        fallbacks: list[str] = []
        candidates: list[DatasetCandidate] = []

        for provider_config in self.config.providers:
            if not provider_config.enabled:
                message = f"Provider {provider_config.name} disabled by configuration"
                self.logger.info(message)
                warnings.append(message)
                continue

            provider = build_provider(provider_config)
            try:
                result = provider.discover(self.config.request)
            except ProviderError as exc:
                message = f"{provider_config.name} failed: {exc}"
                self.logger.warning(message)
                result = ProviderResult(
                    provider=provider_config.name,
                    candidates=(),
                    warnings=(message,),
                    fallbacks=(f"Review {provider_config.name} configuration or use an alternate source.",),
                )

            provider_results.append(result)
            candidates.extend(result.candidates)
            warnings.extend(result.warnings)
            fallbacks.extend(result.fallbacks)
            self.logger.info(
                "Provider %s returned %s candidates",
                provider_config.name,
                len(result.candidates),
            )

        validation_warnings = validate_candidates(candidates, self.config.required_modalities)
        warnings.extend(validation_warnings)

        report = DiscoveryReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            aoi_name=self.config.request.aoi_name,
            start_date=self.config.request.start_date.isoformat(),
            end_date=self.config.request.end_date.isoformat(),
            defaults_used=self.config.defaults,
            candidates=tuple(candidates),
            warnings=tuple(dict.fromkeys(warnings)),
            fallbacks=tuple(dict.fromkeys(fallbacks)),
            provider_results=tuple(provider_results),
        )
        report.write_json(self.config.output_metadata_path)
        self.logger.info("Stage 1 discovery report written to %s", self.config.output_metadata_path)
        return report


def discover_datasets(config_path: str | Path = "configs/discovery.yaml") -> DiscoveryReport:
    """Load config and run Stage 1 discovery."""

    config = load_config(config_path)
    return DiscoveryService(config).run()


def main() -> None:
    """Command-line entrypoint for Stage 1 discovery."""

    report = discover_datasets()
    print(json.dumps(report.to_dict(), indent=2))


def validate_candidates(
    candidates: Iterable[DatasetCandidate],
    required_modalities: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate discovery metadata without downloading or preprocessing data."""

    candidate_list = list(candidates)
    warnings: list[str] = []
    seen_products: set[str] = set()
    modalities = {candidate.modality for candidate in candidate_list}

    for candidate in candidate_list:
        missing = _missing_required_fields(candidate)
        if missing:
            raise ValidationError(
                f"Candidate {candidate.product_id} from {candidate.provider} is missing required fields: {missing}"
            )
        if candidate.product_id in seen_products:
            warnings.append(f"Duplicate product_id discovered: {candidate.product_id}")
        seen_products.add(candidate.product_id)

    for modality in required_modalities:
        if modality not in modalities:
            warnings.append(f"No candidate discovered for required modality: {modality}")

    if not candidate_list:
        warnings.append("No dataset candidates discovered; configure credentials, AOI geometry, or fallback providers")

    return tuple(warnings)


def _missing_required_fields(candidate: DatasetCandidate) -> list[str]:
    required_values = {
        "provider": candidate.provider,
        "source": candidate.source,
        "collection": candidate.collection,
        "product_id": candidate.product_id,
        "modality": candidate.modality,
    }
    return [name for name, value in required_values.items() if not value]


if __name__ == "__main__":
    main()
