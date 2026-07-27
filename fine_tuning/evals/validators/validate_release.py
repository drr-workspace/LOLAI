from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evals.validators import validate_dataset
from generators.checksums import sha256_file, verify_checksums
from generators.release_renderer import SYSTEM_PROMPT


SPLITS = ("train", "valid", "test", "challenge")


def validate(release_dir: Path, config_path: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    results: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = release_dir / f"{split}.jsonl"
        if not path.is_file():
            errors.append(f"{path}: file mancante")
            continue
        result = validate_dataset.validate_split(path)
        results[split] = result
        errors.extend(result["errors"])
    if len(results) == len(SPLITS):
        errors.extend(validate_dataset.verify_no_cross_split_leakage(results))
        errors.extend(validate_dataset.verify_dataset_card(release_dir, results))
    for required in (
        "dataset-card.json",
        "manifest.json",
        "checksums.sha256",
        "build-report.json",
    ):
        if not (release_dir / required).is_file():
            errors.append(f"{release_dir / required}: file mancante")
    checksum_path = release_dir / "checksums.sha256"
    if checksum_path.is_file():
        errors.extend(verify_checksums(checksum_path))
    manifest_path = release_dir / "manifest.json"
    if manifest_path.is_file():
        errors.extend(_validate_manifest(manifest_path, release_dir))
    if config_path is not None and config_path.is_file() and results:
        errors.extend(_validate_distributions(config_path, results))
    expected_system = __import__("hashlib").sha256(
        SYSTEM_PROMPT.encode()
    ).hexdigest()
    for split, result in results.items():
        hashes = result["stats"]["system_hashes"]
        if hashes != {expected_system}:
            errors.append(f"{split}: system prompt non identico al runtime")
    return {
        "valid": not errors,
        "errors": errors,
        "splits": {
            name: validate_dataset.make_serializable_stats(result["stats"])
            for name, result in results.items()
        },
    }


def _validate_manifest(path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [f"{path}: JSON non valido: {error}"]
    files = manifest.get("files")
    if not isinstance(files, dict):
        return [f"{path}: files non valido"]
    for name, item in files.items():
        if not isinstance(item, dict):
            errors.append(f"{path}: files.{name} non valido")
            continue
        target = root / str(item.get("filename"))
        if not target.is_file():
            errors.append(f"{path}: file manifest mancante {target.name}")
        elif item.get("sha256") != sha256_file(target):
            errors.append(f"{path}: checksum manifest errato {target.name}")
        elif item.get("sizeBytes") != target.stat().st_size:
            errors.append(f"{path}: sizeBytes errato {target.name}")
    return errors


def _validate_distributions(
    config_path: Path, results: dict[str, dict[str, Any]]
) -> list[str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for split in SPLITS:
        expected = config.get(f"{split}Examples")
        actual = results.get(split, {}).get("stats", {}).get("examples")
        if actual != expected:
            errors.append(f"{split}: esempi={actual}, attesi={expected}")
    train = results.get("train", {}).get("stats", {})
    total = train.get("examples", 0)
    decisions: Counter[str] = train.get("decisions", Counter())
    if total:
        for decision, bounds in config["decisionDistributionRanges"].items():
            ratio = decisions[decision] / total
            if not bounds["minimum"] <= ratio <= bounds["maximum"]:
                errors.append(
                    f"train {decision}={ratio:.4f} fuori range "
                    f"[{bounds['minimum']}, {bounds['maximum']}]"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=root / "datasets/releases/2.0.0",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "generators/dataset-config.json",
    )
    args = parser.parse_args(argv)
    report = validate(args.release_dir, args.config)
    for error in report["errors"]:
        print(f"ERROR: {error}")
    print(f"Release: {'PASS' if report['valid'] else 'FAIL'}")
    return int(not report["valid"])


if __name__ == "__main__":
    raise SystemExit(main())
