from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from generators.causal_signature import DatasetRecord
from generators.checksums import sha256_file
from generators.release_renderer import SYSTEM_PROMPT
from generators.release_manifest import deterministic_timestamp


def _counts(
    records: Sequence[DatasetRecord], field: str
) -> dict[str, int]:
    if field == "task":
        values = (item.scenario.task for item in records)
    elif field == "decision":
        values = (
            item.oracle_result.decision.decision for item in records
        )
    else:
        values = (item.scenario.output_locale for item in records)
    return dict(sorted(Counter(values).items()))


def build_dataset_card(
    *,
    dataset_version: str,
    schema_version: str,
    ontology_version: str,
    seed: int,
    split_records: Mapping[str, Sequence[DatasetRecord]],
    split_paths: Mapping[str, Path],
) -> dict[str, object]:
    all_records = tuple(
        item for records in split_records.values() for item in records
    )
    splits = {
        name: {
            "filename": path.name,
            "examples": len(split_records[name]),
            "sha256": sha256_file(path),
            "sizeBytes": path.stat().st_size,
            "tasks": _counts(split_records[name], "task"),
            "decisions": _counts(split_records[name], "decision"),
            "locales": _counts(split_records[name], "locale"),
        }
        for name, path in split_paths.items()
    }
    return {
        "datasetName": "LOLAI Advisor Invariant Strategic Dataset",
        "datasetVersion": dataset_version,
        "schemaVersion": schema_version,
        "ontologyVersion": ontology_version,
        "createdAt": deterministic_timestamp(seed),
        "purpose": (
            "Fine-tuning di decisioni strategiche astratte, "
            "evidence-grounded e resistenti a conoscenza volatile."
        ),
        "language": ["it-IT", "en-US"],
        "supportedTasks": sorted(
            {item.scenario.task for item in all_records}
        ),
        "supportedDecisions": [
            "SHOW",
            "SUPPRESS",
            "REQUEST_REFRESH",
        ],
        "format": {
            "type": "JSONL",
            "conversationFormat": "chat",
            "messageOrder": ["system", "user", "assistant"],
            "encoding": "UTF-8",
        },
        "knowledgeBoundary": {
            "policy": "INVARIANT_ONLY",
            "excluded": [
                "Champion, item and rune names",
                "Patch identifiers and volatile statistics",
                "Concrete cooldown and objective timers",
            ],
        },
        "splitPolicy": {
            "train": "Gradient updates",
            "valid": "Model selection",
            "test": "Final held-out evaluation",
            "challenge": "Adversarial evaluation only",
            "leakageRule": (
                "Families, episodes and counterfactual pairs remain cohesive."
            ),
        },
        "systemPrompt": {
            "embeddedInEveryExample": True,
            "mustMatchRuntimePrompt": True,
            "sha256": __import__("hashlib").sha256(
                SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
        },
        "aggregate": {
            "examples": len(all_records),
            "tasks": _counts(all_records, "task"),
            "decisions": _counts(all_records, "decision"),
            "locales": _counts(all_records, "locale"),
        },
        "splits": splits,
        "limitations": [
            "The labels encode deterministic policy assumptions.",
            "Volatile game facts must be supplied by runtime evidence.",
        ],
    }


def write_dataset_card(
    path: Path, card: Mapping[str, object]
) -> None:
    path.write_text(
        json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
