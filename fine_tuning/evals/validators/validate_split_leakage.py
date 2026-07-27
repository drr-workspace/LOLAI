from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from evals.validators.validate_canonical import load_jsonl


SPLITS = ("train", "valid", "test", "challenge")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def release_inputs(
    release_dir: Path,
) -> tuple[dict[str, tuple[str, dict[str, Any], dict[str, Any]]], list[str]]:
    rows: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
    errors: list[str] = []
    for split in SPLITS:
        path = release_dir / f"{split}.jsonl"
        if not path.is_file():
            errors.append(f"{path}: file mancante")
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                    user = json.loads(row["messages"][1]["content"])
                    assistant = json.loads(row["messages"][2]["content"])
                    request_id = user["requestId"]
                except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                    errors.append(f"{path}:{line_number}: {error}")
                    continue
                if request_id in rows:
                    errors.append(f"{path}:{line_number}: requestId duplicato")
                rows[request_id] = (split, user, assistant)
    return rows, errors


def canonical_split_locations(
    canonical_path: Path, release_dir: Path
) -> tuple[dict[str, str], list[str]]:
    scenarios, errors = load_jsonl(canonical_path)
    release, release_errors = release_inputs(release_dir)
    errors.extend(release_errors)
    locations: dict[str, str] = {}
    for scenario in scenarios:
        request_id = scenario.get("input", {}).get("requestId")
        row = release.get(request_id)
        if row is None:
            errors.append(
                f"scenario {scenario.get('scenarioId')}: assente dalla release"
            )
        else:
            locations[str(scenario.get("scenarioId"))] = row[0]
    return locations, errors


def _tokens(value: object, prefix: str = "$") -> frozenset[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"requestId", "actionId", "evidenceId", "entityId"}:
                continue
            result.update(_tokens(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for child in value:
            result.update(_tokens(child, prefix))
    elif isinstance(value, (str, bool)) or value is None:
        result.add(f"{prefix}={value}")
    elif isinstance(value, (int, float)):
        result.add(f"{prefix}~{round(float(value), 1)}")
    return frozenset(result)


def _similarity(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def validate(release_dir: Path, canonical_path: Path) -> list[str]:
    release, errors = release_inputs(release_dir)
    scenarios, canonical_errors = load_jsonl(canonical_path)
    errors.extend(canonical_errors)
    by_request = {
        item.get("input", {}).get("requestId"): item for item in scenarios
    }
    exact_rows: dict[str, str] = {}
    inputs: dict[str, str] = {}
    pairs: dict[str, str] = {}
    signatures: dict[str, str] = {}
    families: defaultdict[str, set[str]] = defaultdict(set)
    episode_splits: defaultdict[str, set[str]] = defaultdict(set)
    pair_splits: defaultdict[str, set[str]] = defaultdict(set)
    token_rows: list[tuple[str, str, frozenset[str]]] = []
    for request_id, (split, user, assistant) in release.items():
        scenario = by_request.get(request_id)
        row_hash = hashlib.sha256(
            _canonical({"user": user, "assistant": assistant}).encode()
        ).hexdigest()
        input_hash = hashlib.sha256(_canonical(user).encode()).hexdigest()
        pair_hash = hashlib.sha256(
            _canonical([user, assistant]).encode()
        ).hexdigest()
        for label, digest, target in (
            ("duplicato esatto", row_hash, exact_rows),
            ("input identico", input_hash, inputs),
            ("coppia identica", pair_hash, pairs),
        ):
            previous = target.setdefault(digest, split)
            if previous != split:
                errors.append(f"{label} tra {previous} e {split}")
        token_rows.append((split, str(request_id), _tokens(user)))
        if scenario is None:
            continue
        signature = scenario.get("causalSignature")
        if isinstance(signature, str):
            previous = signatures.setdefault(signature, split)
            if previous != split:
                errors.append(
                    f"causalSignature sovrapposta tra {previous} e {split}"
                )
        family = scenario.get("familyId")
        if isinstance(family, str):
            families[family].add(split)
        episode = scenario.get("episodeId")
        pair_id = scenario.get("counterfactualPairId")
        if episode:
            episode_splits[str(episode)].add(split)
        if pair_id:
            pair_splits[str(pair_id)].add(split)
    for family, splits in families.items():
        if "train" in splits and "test" in splits:
            errors.append(f"familyId {family!r} presente in train e test")
    for label, groups in (
        ("episodeId", episode_splits),
        ("counterfactualPairId", pair_splits),
    ):
        for identifier, splits in groups.items():
            if len(splits) > 1:
                errors.append(f"{label} {identifier!r} spezzato: {sorted(splits)}")
    comparisons = 0
    for index, (split_a, request_a, tokens_a) in enumerate(token_rows):
        for split_b, request_b, tokens_b in token_rows[index + 1 :]:
            if split_a == split_b:
                continue
            comparisons += 1
            if comparisons > 250_000:
                break
            if _similarity(tokens_a, tokens_b) >= 0.97:
                errors.append(
                    f"near-duplicate leakage {request_a}/{request_b} "
                    f"tra {split_a} e {split_b}"
                )
        if comparisons > 250_000:
            break
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
        "--canonical",
        type=Path,
        default=root / "datasets/canonical/releases/2.0.0/scenarios.jsonl",
    )
    args = parser.parse_args(argv)
    errors = validate(args.release_dir, args.canonical)
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Split leakage: {'FAIL' if errors else 'PASS'}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
