from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from evals.validators import (
    validate_canonical,
    validate_counterfactuals,
    validate_episodes,
    validate_release,
    validate_split_leakage,
)


def run(
    root: Path,
    report_path: Path,
    *,
    release_dir: Path | None = None,
    canonical_dir: Path | None = None,
) -> dict[str, object]:
    resolved_release = (
        release_dir or root / "datasets/releases/2.0.0"
    ).resolve()
    resolved_canonical = (
        canonical_dir or root / "datasets/canonical/releases/2.0.0"
    ).resolve()
    locations, location_errors = validate_split_leakage.canonical_split_locations(
        resolved_canonical / "scenarios.jsonl", resolved_release
    )
    checks: tuple[tuple[str, Callable[[], list[str]]], ...] = (
        (
            "canonical",
            lambda: validate_canonical.validate_file(
                resolved_canonical / "scenarios.jsonl",
                root / "contracts/task-schemas",
            ),
        ),
        (
            "counterfactuals",
            lambda: validate_counterfactuals.validate(
                resolved_canonical / "scenarios.jsonl",
                resolved_canonical / "counterfactual-pairs.jsonl",
                locations,
            ),
        ),
        (
            "episodes",
            lambda: validate_episodes.validate(
                resolved_canonical / "scenarios.jsonl",
                resolved_canonical / "episodes.jsonl",
                locations,
            ),
        ),
        (
            "splitLeakage",
            lambda: validate_split_leakage.validate(
                resolved_release,
                resolved_canonical / "scenarios.jsonl",
            ),
        ),
    )
    results: dict[str, object] = {}
    all_errors = list(location_errors)
    for name, check in checks:
        errors = check()
        results[name] = {"valid": not errors, "errors": errors}
        all_errors.extend(errors)
    release = validate_release.validate(
        resolved_release,
        (
            root / "generators/dataset-config.json"
            if resolved_release.name == "2.0.0"
            else None
        ),
    )
    results["release"] = release
    all_errors.extend(release["errors"])
    report = {
        "datasetVersion": "2.0.0",
        "valid": not all_errors,
        "errorCount": len(all_errors),
        "checks": results,
        "errors": all_errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--release",
        type=Path,
        default=None,
        help="Directory della release da validare.",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=None,
        help="Directory dei sorgenti canonici associati.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "evals/reports/dataset-2.0.0-validation.json",
    )
    args = parser.parse_args(argv)
    report = run(
        args.root.resolve(),
        args.report.resolve(),
        release_dir=args.release,
        canonical_dir=args.canonical,
    )
    print("LOLAI dataset 2.0.0 validation")
    print(f"Esito: {'PASS' if report['valid'] else 'FAIL'}")
    print(f"Errori: {report['errorCount']}")
    for name, result in report["checks"].items():
        print(f"- {name}: {'PASS' if result['valid'] else 'FAIL'}")
    print(f"Report: {args.report}")
    return int(not report["valid"])


if __name__ == "__main__":
    raise SystemExit(main())
