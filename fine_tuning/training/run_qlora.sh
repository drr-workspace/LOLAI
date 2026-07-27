#!/usr/bin/env bash
set -Eeuo pipefail

# LOLAI Advisor — avvio QLoRA con validazione e logging locale.
#
# Eseguire dalla root del repository:
#   bash training/run_qlora.sh
#
# Seguire il log da un altro terminale:
#   tail -f training/logs/latest/full.log
#
# Stato sintetico:
#   cat training/logs/latest/status.txt

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
CONFIG_PATH="${CONFIG_PATH:-training/configs/qwen3-14b-qlora.yaml}"
DATA_DIR="${DATA_DIR:-datasets/releases/2.0.0}"
MODEL_DIR="${MODEL_DIR:-models/base/qwen3-14b-4bit}"
VALIDATOR_PATH="${VALIDATOR_PATH:-evals/validators/validate_dataset.py}"
LOG_ROOT="${LOG_ROOT:-training/logs}"

RUN_ID="$(date '+%Y%m%d-%H%M%S')"
RUN_DIR="$LOG_ROOT/$RUN_ID"
LATEST_LINK="$LOG_ROOT/latest"

FULL_LOG="$RUN_DIR/full.log"
VALIDATION_LOG="$RUN_DIR/dataset-validation.log"
VALIDATION_REPORT="$RUN_DIR/dataset-validation-report.json"
STATUS_FILE="$RUN_DIR/status.txt"
METADATA_FILE="$RUN_DIR/metadata.json"
PID_FILE="$RUN_DIR/pid"
EXIT_CODE_FILE="$RUN_DIR/exit-code.txt"

mkdir -p "$RUN_DIR"
mkdir -p "$LOG_ROOT"

# Aggiorna il collegamento "latest" senza richiedere GNU ln.
rm -f "$LATEST_LINK"
ln -s "$RUN_ID" "$LATEST_LINK"

write_status() {
  local state="$1"
  local detail="${2:-}"
  {
    printf 'state=%s\n' "$state"
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'updated_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'detail=%s\n' "$detail"
    printf 'log=%s\n' "$FULL_LOG"
    printf 'config=%s\n' "$CONFIG_PATH"
    printf 'data=%s\n' "$DATA_DIR"
    printf 'model=%s\n' "$MODEL_DIR"
  } > "$STATUS_FILE"
}

on_interrupt() {
  local exit_code=$?
  write_status "INTERRUPTED" "Training interrotto con codice $exit_code"
  printf '%s\n' "$exit_code" > "$EXIT_CODE_FILE"
  printf '\n[%s] Training interrotto. Exit code: %s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$exit_code" | tee -a "$FULL_LOG"
  exit "$exit_code"
}

trap on_interrupt INT TERM

write_status "PREPARING" "Controllo ambiente e file"

{
  echo "============================================================"
  echo " LOLAI Advisor — Qwen3-14B QLoRA"
  echo "============================================================"
  echo "Run ID:       $RUN_ID"
  echo "Avvio:        $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "Root:         $ROOT_DIR"
  echo "Config:       $CONFIG_PATH"
  echo "Dataset:      $DATA_DIR"
  echo "Modello:      $MODEL_DIR"
  echo "Log completo: $FULL_LOG"
  echo "============================================================"
} | tee "$FULL_LOG"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERRORE: file mancante: $path" | tee -a "$FULL_LOG" >&2
    write_status "FAILED" "File mancante: $path"
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "ERRORE: cartella mancante: $path" | tee -a "$FULL_LOG" >&2
    write_status "FAILED" "Cartella mancante: $path"
    exit 1
  fi
}

require_dir "$VENV_DIR"
require_file "$CONFIG_PATH"
require_dir "$DATA_DIR"
require_dir "$MODEL_DIR"
require_file "$VALIDATOR_PATH"
require_file "$DATA_DIR/train.jsonl"
require_file "$DATA_DIR/valid.jsonl"
require_file "$DATA_DIR/test.jsonl"
require_file "$DATA_DIR/challenge.jsonl"
require_file "$DATA_DIR/dataset-card.json"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

if ! command -v python >/dev/null 2>&1; then
  echo "ERRORE: Python non disponibile nel virtual environment." | tee -a "$FULL_LOG" >&2
  write_status "FAILED" "Python non disponibile"
  exit 1
fi

if ! command -v mlx_lm.lora >/dev/null 2>&1; then
  echo "ERRORE: mlx_lm.lora non trovato. Installa mlx-lm[train]." | tee -a "$FULL_LOG" >&2
  write_status "FAILED" "mlx_lm.lora non disponibile"
  exit 1
fi

PYTHON_VERSION="$(python --version 2>&1)"
MLX_LM_LOCATION="$(command -v mlx_lm.lora)"

python - "$METADATA_FILE" "$RUN_ID" "$CONFIG_PATH" "$DATA_DIR" "$MODEL_DIR" \
  "$PYTHON_VERSION" "$MLX_LM_LOCATION" <<'PY'
from __future__ import annotations

import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    metadata_path,
    run_id,
    config_path,
    data_dir,
    model_dir,
    python_version,
    mlx_lm_location,
) = sys.argv[1:]

metadata = {
    "runId": run_id,
    "startedAt": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "pythonVersion": python_version,
    "mlxLmExecutable": mlx_lm_location,
    "configPath": config_path,
    "dataDirectory": data_dir,
    "modelDirectory": model_dir,
}

Path(metadata_path).write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "$$" > "$PID_FILE"

echo | tee -a "$FULL_LOG"
echo "[1/2] Validazione dataset..." | tee -a "$FULL_LOG"
write_status "VALIDATING" "Validazione dei quattro split"

set +e
python "$VALIDATOR_PATH" \
  --data-dir "$DATA_DIR" \
  --report "$VALIDATION_REPORT" \
  2>&1 | tee "$VALIDATION_LOG" | tee -a "$FULL_LOG"
VALIDATION_EXIT=${PIPESTATUS[0]}
set -e

if [[ "$VALIDATION_EXIT" -ne 0 ]]; then
  echo "ERRORE: validazione dataset fallita." | tee -a "$FULL_LOG" >&2
  printf '%s\n' "$VALIDATION_EXIT" > "$EXIT_CODE_FILE"
  write_status "FAILED" "Validazione dataset fallita"
  exit "$VALIDATION_EXIT"
fi

echo | tee -a "$FULL_LOG"
echo "[2/2] Avvio fine-tuning QLoRA..." | tee -a "$FULL_LOG"
echo "Per seguire l'avanzamento da un altro terminale:" | tee -a "$FULL_LOG"
echo "  tail -f $LATEST_LINK/full.log" | tee -a "$FULL_LOG"
echo | tee -a "$FULL_LOG"

write_status "TRAINING" "Fine-tuning in corso"

# Riduce il buffering di Python, così loss, eval e checkpoint compaiono
# nel file full.log appena vengono stampati da MLX-LM.
export PYTHONUNBUFFERED=1

set +e
mlx_lm.lora \
  --config "$CONFIG_PATH" \
  --data "$DATA_DIR" \
  --model "$MODEL_DIR" \
  2>&1 | tee -a "$FULL_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

printf '%s\n' "$TRAIN_EXIT" > "$EXIT_CODE_FILE"

if [[ "$TRAIN_EXIT" -eq 0 ]]; then
  write_status "COMPLETED" "Fine-tuning terminato correttamente"
  {
    echo
    echo "============================================================"
    echo "Training completato correttamente."
    echo "Fine: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "Run:  $RUN_DIR"
    echo "============================================================"
  } | tee -a "$FULL_LOG"
else
  write_status "FAILED" "mlx_lm.lora terminato con codice $TRAIN_EXIT"
  {
    echo
    echo "============================================================"
    echo "Training fallito."
    echo "Exit code: $TRAIN_EXIT"
    echo "Fine:      $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "Controlla: $FULL_LOG"
    echo "============================================================"
  } | tee -a "$FULL_LOG" >&2
fi

exit "$TRAIN_EXIT"
