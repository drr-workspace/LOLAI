from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "valid", "test", "challenge")
EXPECTED_ROLES = ("system", "user", "assistant")
ALLOWED_DECISIONS = {"SHOW", "SUPPRESS", "REQUEST_REFRESH"}
ALLOWED_TASKS = {
    "COMPOSITION_PLAN",
    "MATCHUP_PLAN",
    "ITEMIZATION_DECISION",
    "MACRO_PRIORITY",
    "THREAT_ASSESSMENT",
    "ADVICE_SUPPRESSION",
}
ALLOWED_LOCALES = {"it-IT", "en-US"}

REQUIRED_USER_FIELDS = {
    "schemaVersion",
    "ontologyVersion",
    "requestId",
    "task",
    "outputLocale",
    "context",
    "evidence",
    "recentAdvice",
    "payload",
}

REQUIRED_OUTPUT_FIELDS = {
    "schemaVersion",
    "decision",
    "category",
    "primaryActionId",
    "alternativeActionIds",
    "priority",
    "confidence",
    "reasonCodes",
    "evidenceIds",
    "message",
    "validForSeconds",
    "recheckTriggers",
}

FORBIDDEN_VOLATILE_KEYS = {
    "patch",
    "patchVersion",
    "riotPatchLabel",
    "ddragonVersion",
    "championName",
    "championId",
    "abilityName",
    "itemName",
    "itemId",
    "runeName",
    "runeId",
    "objectiveSpawnSeconds",
    "objectiveRespawnSeconds",
    "cooldownSeconds",
    "winRate",
    "pickRate",
    "tier",
    "buildName",
}


class ValidationFailure(Exception):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def fail(errors: list[str], location: str, message: str) -> None:
    errors.append(f"{location}: {message}")


def ensure_object(
    value: Any,
    errors: list[str],
    location: str,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        fail(errors, location, "deve essere un oggetto JSON")
        return None
    return value


def scan_forbidden_keys(
    value: Any,
    errors: list[str],
    location: str,
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_VOLATILE_KEYS:
                fail(errors, location, f"campo volatile vietato: {key}")
            scan_forbidden_keys(child, errors, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan_forbidden_keys(child, errors, f"{location}[{index}]")


def validate_context(
    context: Any,
    errors: list[str],
    location: str,
) -> None:
    obj = ensure_object(context, errors, location)
    if obj is None:
        return

    required = {
        "observedAtGameSecond",
        "freshnessSeconds",
        "completeness",
        "uncertainFields",
    }
    missing = required - set(obj)
    if missing:
        fail(errors, location, f"campi mancanti: {sorted(missing)}")

    completeness = obj.get("completeness")
    if not isinstance(completeness, (int, float)) or isinstance(completeness, bool):
        fail(errors, f"{location}.completeness", "deve essere numerico")
    elif not 0 <= float(completeness) <= 1:
        fail(errors, f"{location}.completeness", "deve essere compreso tra 0 e 1")

    for key in ("observedAtGameSecond", "freshnessSeconds"):
        value = obj.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail(errors, f"{location}.{key}", "deve essere un intero non negativo")

    uncertain = obj.get("uncertainFields")
    if not isinstance(uncertain, list):
        fail(errors, f"{location}.uncertainFields", "deve essere una lista")


def validate_evidence(
    evidence: Any,
    errors: list[str],
    location: str,
) -> set[str]:
    if not isinstance(evidence, list):
        fail(errors, location, "deve essere una lista")
        return set()

    ids: set[str] = set()
    for index, item in enumerate(evidence):
        item_location = f"{location}[{index}]"
        obj = ensure_object(item, errors, item_location)
        if obj is None:
            continue

        for key in ("evidenceId", "type", "confidence", "freshnessSeconds", "fact"):
            if key not in obj:
                fail(errors, item_location, f"campo mancante: {key}")

        evidence_id = obj.get("evidenceId")
        if not isinstance(evidence_id, str) or not evidence_id:
            fail(errors, f"{item_location}.evidenceId", "deve essere una stringa non vuota")
        elif evidence_id in ids:
            fail(errors, f"{item_location}.evidenceId", f"duplicato: {evidence_id}")
        else:
            ids.add(evidence_id)

        confidence = obj.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            fail(errors, f"{item_location}.confidence", "deve essere numerico")
        elif not 0 <= float(confidence) <= 1:
            fail(errors, f"{item_location}.confidence", "deve essere compreso tra 0 e 1")

        freshness = obj.get("freshnessSeconds")
        if not isinstance(freshness, int) or isinstance(freshness, bool) or freshness < 0:
            fail(errors, f"{item_location}.freshnessSeconds", "deve essere un intero non negativo")

    return ids


def candidate_ids_from_payload(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return set()

    result: set[str] = set()
    for candidate in candidates:
        if isinstance(candidate, dict):
            action_id = candidate.get("actionId")
            if isinstance(action_id, str) and action_id:
                result.add(action_id)
    return result


def validate_user(
    user: Any,
    errors: list[str],
    location: str,
) -> tuple[str | None, set[str], set[str], str | None]:
    obj = ensure_object(user, errors, location)
    if obj is None:
        return None, set(), set(), None

    missing = REQUIRED_USER_FIELDS - set(obj)
    if missing:
        fail(errors, location, f"campi mancanti: {sorted(missing)}")

    if obj.get("schemaVersion") != "1.0.0":
        fail(errors, f"{location}.schemaVersion", "deve essere 1.0.0")

    if obj.get("ontologyVersion") != "1.0.0":
        fail(errors, f"{location}.ontologyVersion", "deve essere 1.0.0")

    task = obj.get("task")
    if task not in ALLOWED_TASKS:
        fail(errors, f"{location}.task", f"task non valido: {task!r}")
        task = None

    locale = obj.get("outputLocale")
    if locale not in ALLOWED_LOCALES:
        fail(errors, f"{location}.outputLocale", f"locale non valida: {locale!r}")
        locale = None

    request_id = obj.get("requestId")
    if not isinstance(request_id, str) or not request_id:
        fail(errors, f"{location}.requestId", "deve essere una stringa non vuota")
        request_id = None

    validate_context(obj.get("context"), errors, f"{location}.context")
    evidence_ids = validate_evidence(obj.get("evidence"), errors, f"{location}.evidence")

    recent_advice = obj.get("recentAdvice")
    if not isinstance(recent_advice, list):
        fail(errors, f"{location}.recentAdvice", "deve essere una lista")

    payload = obj.get("payload")
    if not isinstance(payload, dict):
        fail(errors, f"{location}.payload", "deve essere un oggetto JSON")

    candidate_ids = candidate_ids_from_payload(payload)

    scan_forbidden_keys(obj, errors, location)
    return task, evidence_ids, candidate_ids, request_id


def validate_assistant(
    assistant: Any,
    input_task: str | None,
    input_locale: str | None,
    evidence_ids: set[str],
    candidate_ids: set[str],
    user_payload: dict[str, Any] | None,
    errors: list[str],
    location: str,
) -> None:
    obj = ensure_object(assistant, errors, location)
    if obj is None:
        return

    missing = REQUIRED_OUTPUT_FIELDS - set(obj)
    if missing:
        fail(errors, location, f"campi mancanti: {sorted(missing)}")

    if obj.get("schemaVersion") != "1.0.0":
        fail(errors, f"{location}.schemaVersion", "deve essere 1.0.0")

    decision = obj.get("decision")
    if decision not in ALLOWED_DECISIONS:
        fail(errors, f"{location}.decision", f"decisione non valida: {decision!r}")

    category = obj.get("category")
    if input_task is not None and category != input_task:
        fail(
            errors,
            f"{location}.category",
            f"deve coincidere con input task {input_task!r}",
        )

    confidence = obj.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        fail(errors, f"{location}.confidence", "deve essere numerico")
    elif not 0 <= float(confidence) <= 1:
        fail(errors, f"{location}.confidence", "deve essere compreso tra 0 e 1")

    message = obj.get("message")
    if not isinstance(message, str):
        fail(errors, f"{location}.message", "deve essere una stringa")
    elif len(message) > 180:
        fail(errors, f"{location}.message", "supera 180 caratteri")

    output_evidence_ids = obj.get("evidenceIds")
    if not isinstance(output_evidence_ids, list):
        fail(errors, f"{location}.evidenceIds", "deve essere una lista")
        output_evidence_ids = []

    invalid_evidence = set(output_evidence_ids) - evidence_ids
    if invalid_evidence:
        fail(
            errors,
            f"{location}.evidenceIds",
            f"riferimenti inesistenti: {sorted(invalid_evidence)}",
        )

    alternatives = obj.get("alternativeActionIds")
    if not isinstance(alternatives, list):
        fail(errors, f"{location}.alternativeActionIds", "deve essere una lista")

    primary_action_id = obj.get("primaryActionId")

    if decision == "SHOW":
        if not isinstance(primary_action_id, str) or not primary_action_id:
            fail(errors, f"{location}.primaryActionId", "SHOW richiede un actionId")
        elif input_task == "ADVICE_SUPPRESSION":
            expected = None
            if isinstance(user_payload, dict):
                candidate_advice = user_payload.get("candidateAdvice")
                if isinstance(candidate_advice, dict):
                    expected = candidate_advice.get("actionId")
            if primary_action_id != expected:
                fail(
                    errors,
                    f"{location}.primaryActionId",
                    f"deve coincidere con candidateAdvice.actionId {expected!r}",
                )
        elif primary_action_id not in candidate_ids:
            fail(
                errors,
                f"{location}.primaryActionId",
                f"actionId non presente nei candidati: {primary_action_id!r}",
            )

        if not message:
            fail(errors, f"{location}.message", "SHOW richiede un messaggio non vuoto")

        if not output_evidence_ids:
            fail(errors, f"{location}.evidenceIds", "SHOW richiede almeno una evidenza")
    elif decision in {"SUPPRESS", "REQUEST_REFRESH"}:
        if primary_action_id is not None:
            fail(errors, f"{location}.primaryActionId", "deve essere null")
        if alternatives not in ([], None):
            fail(errors, f"{location}.alternativeActionIds", "deve essere vuoto")
        if message != "":
            fail(errors, f"{location}.message", "deve essere vuoto")

    valid_for = obj.get("validForSeconds")
    if not isinstance(valid_for, int) or isinstance(valid_for, bool) or valid_for < 0:
        fail(errors, f"{location}.validForSeconds", "deve essere un intero non negativo")

    reason_codes = obj.get("reasonCodes")
    if not isinstance(reason_codes, list) or not reason_codes:
        fail(errors, f"{location}.reasonCodes", "deve essere una lista non vuota")

    triggers = obj.get("recheckTriggers")
    if not isinstance(triggers, list) or not triggers:
        fail(errors, f"{location}.recheckTriggers", "deve essere una lista non vuota")

    if input_locale == "en-US" and isinstance(message, str) and message:
        italian_markers = {
            "scegli",
            "priorità",
            "giocate",
            "ignora",
            "minaccia",
            "squadra",
            "wave è",
        }
        lowered = message.lower()
        if any(marker in lowered for marker in italian_markers):
            fail(errors, f"{location}.message", "sembra non rispettare en-US")

    scan_forbidden_keys(obj, errors, location)


def validate_split(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    stats = {
        "examples": 0,
        "tasks": Counter(),
        "decisions": Counter(),
        "locales": Counter(),
        "request_ids": set(),
        "user_hashes": set(),
        "pair_hashes": set(),
        "system_hashes": set(),
    }

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            location = f"{path.name}:{line_number}"
            line = raw_line.strip()
            if not line:
                fail(errors, location, "riga vuota")
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(errors, location, f"JSONL non valido: {exc}")
                continue

            if not isinstance(row, dict):
                fail(errors, location, "la riga deve essere un oggetto JSON")
                continue

            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                fail(errors, location, "messages deve contenere esattamente 3 elementi")
                continue

            roles = tuple(
                message.get("role") if isinstance(message, dict) else None
                for message in messages
            )
            if roles != EXPECTED_ROLES:
                fail(errors, location, f"ordine ruoli non valido: {roles!r}")
                continue

            contents: list[str] = []
            valid_contents = True
            for index, message in enumerate(messages):
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str):
                    fail(
                        errors,
                        f"{location}.messages[{index}].content",
                        "deve essere una stringa",
                    )
                    valid_contents = False
                    break
                contents.append(content)

            if not valid_contents:
                continue

            system_content, user_content, assistant_content = contents
            stats["system_hashes"].add(
                sha256_bytes(system_content.encode("utf-8"))
            )

            try:
                user = json.loads(user_content)
            except json.JSONDecodeError as exc:
                fail(errors, f"{location}.user", f"JSON non valido: {exc}")
                continue

            try:
                assistant = json.loads(assistant_content)
            except json.JSONDecodeError as exc:
                fail(errors, f"{location}.assistant", f"JSON non valido: {exc}")
                continue

            input_task, evidence_ids, candidate_ids, request_id = validate_user(
                user,
                errors,
                f"{location}.user",
            )

            input_locale = (
                user.get("outputLocale")
                if isinstance(user, dict)
                else None
            )
            payload = (
                user.get("payload")
                if isinstance(user, dict) and isinstance(user.get("payload"), dict)
                else None
            )

            validate_assistant(
                assistant,
                input_task,
                input_locale,
                evidence_ids,
                candidate_ids,
                payload,
                errors,
                f"{location}.assistant",
            )

            stats["examples"] += 1

            if isinstance(user, dict):
                task = user.get("task")
                locale = user.get("outputLocale")
                if isinstance(task, str):
                    stats["tasks"][task] += 1
                if isinstance(locale, str):
                    stats["locales"][locale] += 1

            if isinstance(assistant, dict):
                decision = assistant.get("decision")
                if isinstance(decision, str):
                    stats["decisions"][decision] += 1

            if request_id is not None:
                if request_id in stats["request_ids"]:
                    fail(errors, location, f"requestId duplicato nello split: {request_id}")
                stats["request_ids"].add(request_id)

            canonical_user = canonical_json(user)
            canonical_pair = canonical_json(
                {"user": user, "assistant": assistant}
            )
            stats["user_hashes"].add(
                sha256_bytes(canonical_user.encode("utf-8"))
            )
            stats["pair_hashes"].add(
                sha256_bytes(canonical_pair.encode("utf-8"))
            )

    if len(stats["system_hashes"]) != 1:
        fail(
            errors,
            path.name,
            f"trovati {len(stats['system_hashes'])} system prompt diversi",
        )

    return {"errors": errors, "stats": stats}


def verify_no_cross_split_leakage(
    results: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []

    for index, split_a in enumerate(SPLITS):
        for split_b in SPLITS[index + 1:]:
            stats_a = results[split_a]["stats"]
            stats_b = results[split_b]["stats"]

            request_overlap = stats_a["request_ids"] & stats_b["request_ids"]
            if request_overlap:
                errors.append(
                    f"leakage requestId tra {split_a} e {split_b}: "
                    f"{sorted(request_overlap)[:5]}"
                )

            user_overlap = stats_a["user_hashes"] & stats_b["user_hashes"]
            if user_overlap:
                errors.append(
                    f"leakage input identici tra {split_a} e {split_b}: "
                    f"{len(user_overlap)}"
                )

            pair_overlap = stats_a["pair_hashes"] & stats_b["pair_hashes"]
            if pair_overlap:
                errors.append(
                    f"leakage coppie input/output tra {split_a} e {split_b}: "
                    f"{len(pair_overlap)}"
                )

    return errors


def verify_dataset_card(
    root: Path,
    results: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    card_path = root / "dataset-card.json"

    if not card_path.exists():
        return [f"{card_path}: file mancante"]

    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{card_path}: JSON non valido: {exc}"]

    card_splits = card.get("splits")
    if not isinstance(card_splits, dict):
        return [f"{card_path}: campo splits mancante o non valido"]

    for split in SPLITS:
        split_path = root / f"{split}.jsonl"
        expected = card_splits.get(split)
        if not isinstance(expected, dict):
            errors.append(f"{card_path}: split {split} mancante")
            continue

        actual_hash = sha256_file(split_path)
        if expected.get("sha256") != actual_hash:
            errors.append(
                f"{split_path}: hash non coincide con dataset-card.json"
            )

        actual_examples = results[split]["stats"]["examples"]
        if expected.get("examples") != actual_examples:
            errors.append(
                f"{split_path}: examples={actual_examples}, "
                f"dataset-card={expected.get('examples')}"
            )

    aggregate = card.get("aggregate")
    if isinstance(aggregate, dict):
        expected_total = sum(
            results[split]["stats"]["examples"]
            for split in SPLITS
        )
        if aggregate.get("examples") != expected_total:
            errors.append(
                f"{card_path}: aggregate.examples non coincide con {expected_total}"
            )

    return errors


def make_serializable_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "examples": stats["examples"],
        "tasks": dict(sorted(stats["tasks"].items())),
        "decisions": dict(sorted(stats["decisions"].items())),
        "locales": dict(sorted(stats["locales"].items())),
        "uniqueRequestIds": len(stats["request_ids"]),
        "uniqueInputs": len(stats["user_hashes"]),
        "uniquePairs": len(stats["pair_hashes"]),
        "systemPromptHashes": sorted(stats["system_hashes"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida la release del dataset LOLAI Advisor.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/releases/1.0.0"),
        help="Cartella contenente train.jsonl, valid.jsonl, test.jsonl, "
             "challenge.jsonl e dataset-card.json.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Percorso opzionale per salvare il report JSON.",
    )
    args = parser.parse_args()

    root = args.data_dir.resolve()
    missing_files = [
        root / f"{split}.jsonl"
        for split in SPLITS
        if not (root / f"{split}.jsonl").exists()
    ]

    if missing_files:
        for path in missing_files:
            print(f"ERRORE: file mancante: {path}", file=sys.stderr)
        return 1

    results: dict[str, dict[str, Any]] = {}
    all_errors: list[str] = []

    for split in SPLITS:
        result = validate_split(root / f"{split}.jsonl")
        results[split] = result
        all_errors.extend(result["errors"])

    all_errors.extend(verify_no_cross_split_leakage(results))
    all_errors.extend(verify_dataset_card(root, results))

    report = {
        "datasetDirectory": str(root),
        "valid": not all_errors,
        "splits": {
            split: make_serializable_stats(results[split]["stats"])
            for split in SPLITS
        },
        "errors": all_errors,
    }

    if args.report is not None:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if all_errors:
        print("VALIDAZIONE FALLITA")
        for error in all_errors:
            print(f"- {error}")
        return 1

    print("VALIDAZIONE COMPLETATA")
    total = 0
    for split in SPLITS:
        stats = report["splits"][split]
        total += stats["examples"]
        print(
            f"- {split}: {stats['examples']} esempi, "
            f"{stats['uniqueInputs']} input unici, "
            f"{stats['uniquePairs']} coppie uniche"
        )
    print(f"Totale: {total} esempi")
    print("Nessun duplicato esatto tra gli split.")
    print("Hash e conteggi coerenti con dataset-card.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
