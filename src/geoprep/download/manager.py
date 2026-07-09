"""Stage 2: resumable download manager with manifests and checksums."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from geoprep.core.exceptions import DownloadError
from geoprep.core.io import sha256_file, write_json
from geoprep.core.logging import get_logger
from geoprep.core.models import DownloadArtifact


class DownloadManager:
    """Download discovered raw assets without preprocessing them."""

    def __init__(self, output_dir: str = "outputs/raw", retries: int = 3, logger_name: str = "geoprep.download") -> None:
        self.output_dir = Path(output_dir)
        self.retries = retries
        self.logger = get_logger(logger_name)

    def download_candidates(self, candidates: list[dict[str, Any]], manifest_path: str = "outputs/manifests/download_manifest.json") -> list[DownloadArtifact]:
        """Download all candidates that have URLs and write a manifest."""

        artifacts: list[DownloadArtifact] = []
        for candidate in candidates:
            artifact = self.download_candidate(candidate)
            artifacts.append(artifact)
        write_json(manifest_path, {"artifacts": [artifact.to_dict() for artifact in artifacts]})
        return artifacts

    def download_candidate(self, candidate: dict[str, Any]) -> DownloadArtifact:
        """Download one candidate or record an unavailable catalog-level artifact."""

        product_id = str(candidate["product_id"])
        modality = str(candidate["modality"])
        url = candidate.get("access_url")
        safe_name = product_id.replace("/", "_").replace(":", "_")
        destination = self.output_dir / modality / f"{safe_name}.dat"
        destination.parent.mkdir(parents=True, exist_ok=True)

        if not url:
            metadata_path = destination.with_suffix(".metadata.json")
            write_json(metadata_path, {"candidate": candidate, "status": "metadata_only_no_download_url"})
            checksum = sha256_file(metadata_path)
            return DownloadArtifact(product_id, modality, None, str(metadata_path), checksum, metadata_path.stat().st_size, False, {"status": "metadata_only"})

        resumed = destination.exists()
        for attempt in range(1, self.retries + 1):
            try:
                self._download_url(str(url), destination, resume=resumed)
                checksum = sha256_file(destination)
                expected = candidate.get("checksum_sha256")
                if expected and expected != checksum:
                    raise DownloadError(f"Checksum mismatch for {product_id}: expected {expected}, got {checksum}")
                return DownloadArtifact(product_id, modality, str(url), str(destination), checksum, destination.stat().st_size, resumed, {"attempts": attempt})
            except (OSError, urllib.error.URLError, DownloadError) as exc:
                self.logger.warning("download attempt failed", extra={"stage": "stage_2_download"})
                if attempt == self.retries:
                    raise DownloadError(f"Failed to download {product_id} after {self.retries} attempts: {exc}") from exc
                time.sleep(0.1 * attempt)
        raise DownloadError(f"Unreachable download failure for {product_id}")

    def _download_url(self, url: str, destination: Path, resume: bool) -> None:
        headers: dict[str, str] = {}
        mode = "wb"
        if resume and destination.exists():
            existing_size = destination.stat().st_size
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"
                mode = "ab"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response, destination.open(mode + "") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
