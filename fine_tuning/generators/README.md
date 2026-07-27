# LOLAI dataset builder

Il builder crea release riproducibili a partire da contratti, ontologia,
policy, template e oracle deterministico. Non usa un LLM e non modifica le
release precedenti.

Da `fine_tuning/`:

```bash
python -m generators.build_dataset \
  --config generators/dataset-config.json \
  --output datasets/releases/2.0.0
```

Opzioni:

- `--dry-run`: esegue preflight, generazione, firma, deduplicazione e split
  senza scrivere file.
- `--limit-per-task N`: limita la build per smoke test; in questa modalità i
  range percentuali delle decisioni sono riportati ma non bloccanti.
- `--force`: sostituisce la release di destinazione e i relativi sorgenti
  canonici. Senza questa opzione una destinazione esistente causa errore.

La scrittura avviene in directory di staging e viene pubblicata solo dopo la
validazione. La release contiene i quattro split JSONL, dataset card, manifest,
checksum e report. I sorgenti canonici sono salvati sotto
`datasets/canonical/releases/<version>/`.

Gli snapshot realistici vengono inclusi soltanto quando provengono dai file
configurati e hanno `reviewStatus` uguale ad `APPROVED`. Se non sono
disponibili, la quota viene assegnata deterministicamente agli scenari
sintetici e la variazione è registrata nel report.
