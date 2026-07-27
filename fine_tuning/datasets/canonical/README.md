# Canonical scenario workspace

This directory is reserved for abstract canonical scenarios produced from
validated runtime snapshots or human annotations.

Generated review queues, append-only audit logs and approved exports must not
contain champion, item, rune or patch knowledge. Technical provenance is kept
beside canonical records for traceability and is never rendered into model
messages.

Typical review workflow:

```bash
python3 -m fine_tuning.generators.review_cli list
python3 -m fine_tuning.generators.review_cli show SCENARIO_ID
python3 -m fine_tuning.generators.review_cli approve SCENARIO_ID --reviewer REVIEWER_ID
python3 -m fine_tuning.generators.review_cli export-approved approved.jsonl
```

Queue and audit paths can be overridden with `--queue` and `--audit`.
