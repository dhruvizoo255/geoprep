"""Stage 4: temporal, CRS, spatial, and resolution alignment.

Supports three cases, since AlignedObservation.optical and .sar are both
Optional by design:
  1. Optical + SAR both present -> paired within the temporal window
     (original behaviour, unchanged).
  2. Optical only (e.g. pure Sentinel-2 workflows, no SAR available) ->
     one optical-only AlignedObservation per optical scene.
  3. SAR only -> one SAR-only AlignedObservation per SAR scene.

Previously, if `sar_scenes` was empty, every optical scene was silently
dropped (aligned == [] regardless of how much optical data was supplied).
That contradicted the AlignedObservation model, which already allows a
None sar/optical field, so this was a real gap rather than the intended
"AlignmentError when nothing matches" behaviour.
"""

from __future__ import annotations

from datetime import date

from geoprep.core.array_ops import nearest_resample
from geoprep.core.exceptions import AlignmentError
from geoprep.core.logging import get_logger
from geoprep.core.models import AlignedObservation, RasterScene


class AlignmentService:
    """Align optical and SAR observations for multimodal learning."""

    def __init__(self, temporal_window_days: int = 3, logger_name: str = "geoprep.alignment") -> None:
        self.temporal_window_days = temporal_window_days
        self.logger = get_logger(logger_name)

    def align(self, optical_scenes: list[RasterScene], sar_scenes: list[RasterScene]) -> list[AlignedObservation]:
        """Pair optical and SAR scenes within the configured temporal window.

        Any optical scene with no SAR match (including when sar_scenes is
        empty) becomes an optical-only observation. Any SAR scene left
        unmatched (including when optical_scenes is empty) becomes a
        SAR-only observation. AlignmentError is only raised if both lists
        are non-empty and truly nothing -- paired or single-modality --
        could be produced, which in practice cannot happen as long as at
        least one scene was supplied.
        """

        self.logger.info("Aligning observations", extra={"stage": "align"})

        aligned: list[AlignedObservation] = []
        used_sar: set[str] = set()

        for optical in sorted(optical_scenes, key=lambda scene: scene.acquisition_date):
            candidates = [scene for scene in sar_scenes if scene.scene_id not in used_sar]
            sar = self._nearest_scene(optical, candidates)

            if sar is None:
                aligned.append(
                    AlignedObservation(
                        observation_id=f"{optical.scene_id}__optical_only",
                        acquisition_date=optical.acquisition_date,
                        optical=optical,
                        sar=None,
                        crs=optical.crs,
                        resolution_m=optical.resolution_m,
                        metadata={**optical.metadata, "modality_pairing": "optical_only"},
                    )
                )
                continue

            used_sar.add(sar.scene_id)
            matched_sar = self._match_grid(sar, optical.crs, optical.resolution_m, optical.band_shape())
            aligned.append(
                AlignedObservation(
                    observation_id=f"{optical.scene_id}__{sar.scene_id}",
                    acquisition_date=optical.acquisition_date,
                    optical=optical,
                    sar=matched_sar,
                    crs=optical.crs,
                    resolution_m=optical.resolution_m,
                    metadata={
                        "temporal_delta_days": abs(
                            _to_date(optical.acquisition_date).toordinal() - _to_date(sar.acquisition_date).toordinal()
                        ),
                        "modality_pairing": "optical_sar",
                    },
                )
            )

        for sar in sar_scenes:
            if sar.scene_id in used_sar:
                continue
            aligned.append(
                AlignedObservation(
                    observation_id=f"{sar.scene_id}__sar_only",
                    acquisition_date=sar.acquisition_date,
                    optical=None,
                    sar=sar,
                    crs=sar.crs,
                    resolution_m=sar.resolution_m,
                    metadata={**sar.metadata, "modality_pairing": "sar_only"},
                )
            )

        if (optical_scenes or sar_scenes) and not aligned:
            raise AlignmentError("No optical or SAR observations could be produced")

        return aligned

    def _nearest_scene(self, reference: RasterScene, candidates: list[RasterScene]) -> RasterScene | None:
        reference_date = _to_date(reference.acquisition_date)
        best: tuple[int, RasterScene] | None = None
        for candidate in candidates:
            delta = abs(reference_date.toordinal() - _to_date(candidate.acquisition_date).toordinal())
            if delta <= self.temporal_window_days and (best is None or delta < best[0]):
                best = (delta, candidate)
        return best[1] if best else None

    def _match_grid(self, scene: RasterScene, crs: str, resolution_m: float, shape: tuple[int, int]) -> RasterScene:
        bands = {name: nearest_resample(values, *shape) for name, values in scene.bands.items()}
        return RasterScene(scene.scene_id, scene.modality, scene.acquisition_date, crs, resolution_m, bands, scene.cloud_mask, {**scene.metadata, "aligned_to_crs": crs})


def _to_date(value: str) -> date:
    return date.fromisoformat(value[:10])
