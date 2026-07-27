from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/base/qwen3-14b-4bit"
DEFAULT_ADAPTER = ROOT / "models/adapters/qwen3-14b-lolai-v2.0.0"
CODE_FENCE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)
THINKING_BLOCK = re.compile(
    r"<think>.*?</think>",
    flags=re.IGNORECASE | re.DOTALL,
)


class PredictionError(RuntimeError):
    pass


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = THINKING_BLOCK.sub("", text).strip()
    candidates = [
        match.group(1).strip() for match in CODE_FENCE.finditer(cleaned)
    ]
    candidates.append(cleaned)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def dataset_rows(
    path: Path,
    *,
    start_line: int = 1,
    limit: int | None = None,
) -> Iterator[tuple[int, tuple[dict[str, str], ...], dict[str, Any]]]:
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number < start_line:
                continue
            if limit is not None and emitted >= limit:
                break
            try:
                row = json.loads(line)
                messages = row["messages"]
                system = messages[0]
                user = messages[1]
                expected = messages[2]
                user_json = json.loads(user["content"])
            except (
                KeyError,
                IndexError,
                TypeError,
                json.JSONDecodeError,
            ) as error:
                raise PredictionError(
                    f"{path}:{line_number}: riga dataset non valida: {error}"
                ) from error
            if (
                system.get("role") != "system"
                or user.get("role") != "user"
                or expected.get("role") != "assistant"
            ):
                raise PredictionError(
                    f"{path}:{line_number}: sequenza ruoli non valida"
                )
            prompt_messages = (
                {
                    "role": "system",
                    "content": str(system["content"]),
                },
                {
                    "role": "user",
                    "content": str(user["content"]),
                },
            )
            emitted += 1
            yield line_number, prompt_messages, user_json


def completed_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise PredictionError(
                    f"{path}:{line_number}: output parziale non valido"
                ) from error
            if not isinstance(value, dict) or "prediction" not in value:
                raise PredictionError(
                    f"{path}:{line_number}: record prediction non valido"
                )
            count += 1
    return count


def prediction_record(
    *,
    raw_response: str,
    dataset_line: int,
    user_input: Mapping[str, Any],
    elapsed_seconds: float,
    model_path: Path,
    adapter_path: Path,
) -> dict[str, Any]:
    prediction = extract_json_object(raw_response)
    return {
        "prediction": prediction,
        "metadata": {
            "adapter": str(adapter_path),
            "datasetLine": dataset_line,
            "elapsedSeconds": round(elapsed_seconds, 6),
            "jsonValid": prediction is not None,
            "model": str(model_path),
            "rawResponse": raw_response,
            "requestId": user_input.get("requestId"),
            "task": user_input.get("task"),
        },
    }


def generate_predictions(
    *,
    dataset: Path,
    output: Path,
    model_path: Path,
    adapter_path: Path,
    generate_one: Callable[[Sequence[Mapping[str, str]]], str],
    start_line: int = 1,
    limit: int | None = None,
    resume: bool = False,
    force: bool = False,
    progress_every: int = 10,
) -> int:
    if output.exists() and not resume and not force:
        raise PredictionError(
            f"output già esistente: {output}; usare --resume o --force"
        )
    completed = completed_rows(output) if resume else 0
    selected_rows = dataset_rows(
        dataset, start_line=start_line, limit=limit
    )
    for _ in range(completed):
        try:
            next(selected_rows)
        except StopIteration as error:
            raise PredictionError(
                "l'output contiene più righe dell'intervallo dataset"
            ) from error
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    generated = 0
    with output.open(mode, encoding="utf-8") as handle:
        for dataset_line, messages, user_input in selected_rows:
            started = time.perf_counter()
            raw_response = generate_one(messages)
            elapsed = time.perf_counter() - started
            record = prediction_record(
                raw_response=raw_response,
                dataset_line=dataset_line,
                user_input=user_input,
                elapsed_seconds=elapsed,
                model_path=model_path,
                adapter_path=adapter_path,
            )
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            handle.flush()
            generated += 1
            total_done = completed + generated
            if total_done % progress_every == 0:
                print(
                    f"[predictions] {total_done} righe completate; "
                    f"ultima riga dataset={dataset_line}",
                    flush=True,
                )
    return completed + generated


def mlx_generator(
    *,
    model_path: Path,
    adapter_path: Path,
    max_tokens: int,
    seed: int,
) -> Callable[[Sequence[Mapping[str, str]]], str]:
    try:
        import mlx.core as mx
        from mlx_lm import generate, load
    except (ImportError, RuntimeError) as error:
        raise PredictionError(
            "mlx-lm non disponibile nell'ambiente attivo"
        ) from error
    mx.random.seed(seed)
    model, tokenizer = load(
        str(model_path),
        adapter_path=str(adapter_path),
    )

    def generate_one(messages: Sequence[Mapping[str, str]]) -> str:
        prompt = tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

    return generate_one


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("deve essere >= 1")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera prediction JSONL con MLX-LM."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--start-line", type=_positive, default=1)
    parser.add_argument("--limit", type=_positive)
    parser.add_argument("--max-tokens", type=_positive, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=_positive, default=10)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        for path, label in (
            (args.dataset, "dataset"),
            (args.model, "modello"),
            (args.adapter, "adapter"),
        ):
            if not path.exists():
                raise PredictionError(f"{label} mancante: {path}")
        generate_one = mlx_generator(
            model_path=args.model,
            adapter_path=args.adapter,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        count = generate_predictions(
            dataset=args.dataset,
            output=args.output,
            model_path=args.model,
            adapter_path=args.adapter,
            generate_one=generate_one,
            start_line=args.start_line,
            limit=args.limit,
            resume=args.resume,
            force=args.force,
            progress_every=args.progress_every,
        )
    except (OSError, PredictionError, ValueError) as error:
        print(f"GENERAZIONE FALLITA: {error}")
        return 1
    print(f"Prediction completate: {count}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
