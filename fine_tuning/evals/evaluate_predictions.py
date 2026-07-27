from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals.metrics import (
    abstention_metrics,
    calibration_metrics,
    counterfactual_metrics,
    decision_metrics,
    grounding_metrics,
    robustness_metrics,
    structural_metrics,
)
from evals.validators.validate_contracts import (
    build_registry,
    build_validators,
    load_schemas,
)
from generators.ontology_registry import OntologyRegistry


ROOT = Path(__file__).resolve().parents[1]


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            try:
                user = json.loads(row["messages"][1]["content"])
                expected = json.loads(row["messages"][2]["content"])
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            rows.append({"input": user, "expected": expected})
    return rows


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                predictions.append(
                    {
                        "prediction": None,
                        "metadata": {},
                        "predictionValid": False,
                    }
                )
                continue
            metadata: Mapping[str, Any] = {}
            value: object = raw
            if isinstance(raw, dict) and "prediction" in raw:
                metadata_value = raw.get("metadata", {})
                metadata = (
                    metadata_value
                    if isinstance(metadata_value, dict)
                    else {}
                )
                value = raw["prediction"]
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = None
            predictions.append(
                {
                    "prediction": value if isinstance(value, dict) else None,
                    "metadata": dict(metadata),
                    "predictionValid": isinstance(value, dict),
                    "lineNumber": line_number,
                }
            )
    return predictions


def build_samples(
    dataset: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(dataset) != len(predictions):
        raise ValueError(
            f"righe dataset/predictions diverse: "
            f"{len(dataset)}/{len(predictions)}"
        )
    return [
        {**expected, **prediction}
        for expected, prediction in zip(dataset, predictions)
    ]


def evaluate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    schemas = load_schemas(ROOT / "contracts/task-schemas")
    registry = build_registry(schemas)
    output_validator = build_validators(schemas, registry)[
        "advisor-output.schema.json"
    ]
    ontology = OntologyRegistry(ROOT / "ontology")
    metrics = {
        "structural": structural_metrics.compute(
            samples,
            schema_valid=lambda value: not tuple(
                output_validator.iter_errors(value)
            ),
        ),
        "decision": decision_metrics.compute(samples),
        "abstention": abstention_metrics.compute(samples),
        "grounding": grounding_metrics.compute(
            samples,
            allowed_reason_codes=set(ontology.ids("reason-codes")),
        ),
        "counterfactual": counterfactual_metrics.compute(samples),
        "robustness": robustness_metrics.compute(samples),
        "calibration": calibration_metrics.compute(samples),
    }
    return {"examples": len(samples), "metrics": metrics}


def _flatten(value: object) -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, (int, float)) and not isinstance(child, bool):
                result[key] = float(child)
            else:
                result.update(_flatten(child))
    return result


def apply_quality_gates(
    report: Mapping[str, Any], gates: Mapping[str, Any]
) -> tuple[dict[str, object], ...]:
    values = _flatten(report["metrics"])
    results: list[dict[str, object]] = []
    for metric, rule in gates.items():
        actual = values.get(metric)
        operator = rule["operator"]
        threshold = float(rule["value"])
        passed = (
            actual is not None
            and (
                actual >= threshold
                if operator == "minimum"
                else actual <= threshold
            )
        )
        results.append(
            {
                "metric": metric,
                "operator": operator,
                "threshold": threshold,
                "actual": actual,
                "passed": passed,
            }
        )
    return tuple(results)


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# LOLAI model evaluation",
        "",
        f"Examples: {report['examples']}",
        "",
        "## Quality gates",
        "",
        "| Metric | Actual | Gate | Result |",
        "|---|---:|---:|---|",
    ]
    for gate in report["qualityGates"]:
        actual = gate["actual"]
        rendered = "n/a" if actual is None else f"{actual:.4f}"
        symbol = ">=" if gate["operator"] == "minimum" else "<="
        lines.append(
            f"| {gate['metric']} | {rendered} | "
            f"{symbol} {gate['threshold']:.4f} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Overall: **{'PASS' if report['qualityGatesPassed'] else 'FAIL'}**",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--quality-gates",
        type=Path,
        default=ROOT / "evals/quality-gates-standard.json",
    )
    args = parser.parse_args(argv)
    try:
        samples = build_samples(
            _load_dataset(args.dataset), _load_predictions(args.predictions)
        )
        report = evaluate(samples)
        gates = json.loads(args.quality_gates.read_text(encoding="utf-8"))
        gate_results = apply_quality_gates(report, gates)
        report["qualityGates"] = gate_results
        report["qualityGatesPassed"] = all(
            item["passed"] for item in gate_results
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        args.output.with_suffix(".md").write_text(
            markdown_report(report), encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"VALUTAZIONE FALLITA: {error}")
        return 1
    print(
        f"Quality gates: {'PASS' if report['qualityGatesPassed'] else 'FAIL'}"
    )
    print(f"JSON: {args.output}")
    print(f"Markdown: {args.output.with_suffix('.md')}")
    return int(not report["qualityGatesPassed"])


if __name__ == "__main__":
    raise SystemExit(main())
