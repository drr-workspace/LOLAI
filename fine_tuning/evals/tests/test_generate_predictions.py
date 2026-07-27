from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.generate_predictions import (
    PredictionError,
    completed_rows,
    dataset_rows,
    extract_json_object,
    generate_predictions,
)


def _write_dataset(path: Path, count: int = 3) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            row = {
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "requestId": f"request_{index}",
                                "task": "MACRO_PRIORITY",
                            }
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"decision": "SHOW"}),
                    },
                ]
            }
            handle.write(json.dumps(row) + "\n")


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ('{"decision":"SHOW"}', {"decision": "SHOW"}),
        (
            '```json\n{"decision":"SUPPRESS"}\n```',
            {"decision": "SUPPRESS"},
        ),
        (
            '<think>private</think> result: {"decision":"REQUEST_REFRESH"}',
            {"decision": "REQUEST_REFRESH"},
        ),
    ),
)
def test_extract_json_object(
    text: str, expected: dict[str, str]
) -> None:
    assert extract_json_object(text) == expected


def test_invalid_response_returns_none() -> None:
    assert extract_json_object("not json") is None


def test_dataset_rows_excludes_expected_assistant(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)

    rows = list(dataset_rows(dataset, start_line=2, limit=1))

    assert rows[0][0] == 2
    assert [message["role"] for message in rows[0][1]] == [
        "system",
        "user",
    ]
    assert rows[0][2]["requestId"] == "request_1"


def test_generation_writes_evaluator_compatible_rows(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_dataset(dataset, count=2)

    count = generate_predictions(
        dataset=dataset,
        output=output,
        model_path=Path("model"),
        adapter_path=Path("adapter"),
        generate_one=lambda _: '```json\n{"decision":"SHOW"}\n```',
        progress_every=10,
    )
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]

    assert count == 2
    assert rows[0]["prediction"] == {"decision": "SHOW"}
    assert rows[0]["metadata"]["datasetLine"] == 1
    assert rows[0]["metadata"]["jsonValid"] is True


def test_resume_skips_completed_dataset_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_dataset(dataset, count=3)
    output.write_text(
        json.dumps(
            {
                "prediction": {"decision": "SHOW"},
                "metadata": {"datasetLine": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count = generate_predictions(
        dataset=dataset,
        output=output,
        model_path=Path("model"),
        adapter_path=Path("adapter"),
        generate_one=lambda _: '{"decision":"SUPPRESS"}',
        resume=True,
        progress_every=10,
    )
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]

    assert count == 3
    assert [row["metadata"]["datasetLine"] for row in rows] == [1, 2, 3]


def test_existing_output_requires_explicit_mode(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "predictions.jsonl"
    _write_dataset(dataset, count=1)
    output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(PredictionError, match="già esistente"):
        generate_predictions(
            dataset=dataset,
            output=output,
            model_path=Path("model"),
            adapter_path=Path("adapter"),
            generate_one=lambda _: "{}",
        )


def test_completed_rows_rejects_truncated_output(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text('{"prediction":{}}\n{', encoding="utf-8")

    with pytest.raises(PredictionError, match="parziale"):
        completed_rows(output)
