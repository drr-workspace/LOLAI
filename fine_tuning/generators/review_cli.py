from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from generators.review_queue import (
    ReviewQueue,
    ReviewQueueError,
)


CANONICAL_DIR = Path(__file__).resolve().parents[1] / "datasets" / "canonical"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review abstract canonical LOLAI scenarios."
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=CANONICAL_DIR / "review-queue.jsonl",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=CANONICAL_DIR / "review-audit.jsonl",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    list_command = commands.add_parser("list")
    list_command.add_argument("--status")

    show = commands.add_parser("show")
    show.add_argument("scenario_id")

    for name in ("approve", "reject"):
        command = commands.add_parser(name)
        command.add_argument("scenario_id")
        command.add_argument("--reviewer", required=True)

    edit = commands.add_parser("edit-expected")
    edit.add_argument("scenario_id")
    edit.add_argument("field")
    edit.add_argument("value", help="JSON-encoded replacement value")
    edit.add_argument("--reviewer", required=True)

    note = commands.add_parser("add-note")
    note.add_argument("scenario_id")
    note.add_argument("note")
    note.add_argument("--reviewer", required=True)

    export = commands.add_parser("export-approved")
    export.add_argument("output", type=Path)
    return parser


def _summary(entry: dict[str, Any]) -> dict[str, object]:
    scenario = entry["scenario"]
    expected = entry["expectedOutput"]
    return {
        "scenarioId": scenario["scenarioId"],
        "task": scenario["task"],
        "decision": expected["decision"],
        "reviewStatus": entry["reviewStatus"],
        "reviewPriority": entry["reviewPriority"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    queue = ReviewQueue(args.queue, args.audit)
    try:
        if args.command == "list":
            result = [
                _summary(dict(entry))
                for entry in queue.list(status=args.status)
            ]
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "show":
            print(
                json.dumps(
                    queue.show(args.scenario_id),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command in {"approve", "reject"}:
            status = (
                "APPROVED" if args.command == "approve" else "REJECTED"
            )
            result = queue.set_status(
                args.scenario_id, status, args.reviewer
            )
            print(json.dumps(_summary(dict(result)), ensure_ascii=False))
        elif args.command == "edit-expected":
            try:
                value = json.loads(args.value)
            except json.JSONDecodeError as error:
                raise ReviewQueueError(
                    f"value non è JSON valido: {error.msg}"
                ) from error
            result = queue.edit_expected(
                args.scenario_id,
                args.field,
                value,
                args.reviewer,
            )
            print(json.dumps(_summary(dict(result)), ensure_ascii=False))
        elif args.command == "add-note":
            result = queue.add_note(
                args.scenario_id, args.note, args.reviewer
            )
            print(json.dumps(_summary(dict(result)), ensure_ascii=False))
        elif args.command == "export-approved":
            count = queue.export_approved(args.output)
            print(f"Esportati {count} scenari approvati.")
        else:
            raise ReviewQueueError(f"comando non supportato: {args.command}")
    except (OSError, ReviewQueueError, json.JSONDecodeError) as error:
        print(f"Errore review: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
