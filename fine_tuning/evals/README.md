# LOLAI model evaluation

Eseguire i comandi dalla cartella `fine_tuning/` usando la forma modulo.

## Validation standard

Generare le predizioni con un limite sufficiente per gli scenari adversariali
con liste lunghe:

```bash
python -m evals.generate_predictions \
  --model models/base/qwen3-14b-4bit \
  --adapter evals/checkpoints/1300 \
  --dataset datasets/releases/2.0.0/valid.jsonl \
  --output evals/reports/checkpoint-selection/1300-predictions.jsonl \
  --max-tokens 1024 \
  --seed 42
```

Valutare decisione, struttura e grounding:

```bash
python -m evals.evaluate_predictions \
  --dataset datasets/releases/2.0.0/valid.jsonl \
  --predictions evals/reports/checkpoint-selection/1300-predictions.jsonl \
  --output evals/reports/checkpoint-selection/1300-metrics.json
```

Il comando usa `quality-gates-standard.json`. Le metriche paired prive di
esempi applicabili sono riportate come `null`, con conteggio coppie uguale a
zero, e non fanno parte dei gate standard.

## Benchmark paired

Robustezza e consistenza controfattuale richiedono un dataset di benchmark nel
quale ogni prediction abbia metadata come `robustnessPairId`,
`robustnessType`, `counterfactualPairId` ed `expectedEffect`.

Per quel benchmark usare:

```bash
python -m evals.evaluate_predictions \
  --dataset path/to/paired.jsonl \
  --predictions path/to/paired-predictions.jsonl \
  --output evals/reports/paired-metrics.json \
  --quality-gates evals/quality-gates-paired.json
```

`quality-gates.json` conserva la configurazione completa combinata per suite
che includono sia esempi standard sia coppie annotate.
