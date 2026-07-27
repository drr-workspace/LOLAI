from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from generators.checksums import sha256_file


def build_manifest(
    *,
    dataset_version: str,
    schema_version: str,
    ontology_version: str,
    seed: int,
    files: Mapping[str, Path],
    build_report: Mapping[str, object],
) -> dict[str, object]:
    return {
        "datasetVersion": dataset_version,
        "schemaVersion": schema_version,
        "ontologyVersion": ontology_version,
        "seed": seed,
        "createdAt": deterministic_timestamp(seed),
        "reproducible": True,
        "files": {
            name: {
                "filename": path.name,
                "sha256": sha256_file(path),
                "sizeBytes": path.stat().st_size,
            }
            for name, path in sorted(files.items())
        },
        "build": {
            "examples": build_report["counts"],
            "uniqueCausalSignatures": build_report[
                "uniqueCausalSignatures"
            ],
            "duplicatesRemoved": build_report["duplicatesRemoved"],
        },
    }


def write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def deterministic_timestamp(seed: int) -> str:
    value = str(seed)
    if len(value) == 8 and value.isdigit():
        try:
            return datetime.strptime(value, "%Y%m%d").replace(
                tzinfo=timezone.utc
            ).isoformat()
        except ValueError:
            pass
    return datetime.fromtimestamp(0, tz=timezone.utc).isoformat()
