# Entity extraction - adding your own task

This directory provides an example of adding your own entity extraction task and configuring it.

## Folder structure

- `run.py` – runnable script.
- `tasks.json` – task configuration (points to data + metric config).
- `data/`
  - `redocred_small_extracts.jsonl` – documents / extracts (one JSON object per line) consumed as input.
  - `redocred_small_entities.jsonl` – ground‑truth extracted entities (aligned line‑wise with extracts).
  - `property_schema.json` – property schema used in ground truth
  - `metrics_config.json` – metric configuration (overrides defaults).
  - `predictions.jsonl` – sample model output (for demonstration).

## Quick Start

* install the package (in repo root):

```bash
uv sync
```

## Adding a New Extraction Task

1. Prepare `extracts.jsonl` (similar to `redocred_small _extracts.jsonl`)
2. Prepare aligned `ground_truth_entities.jsonl` (each line: JSON array of entity objects).
3. Create / adapt a `property_schema.json` for your ground truth entities.
4. Create a metric config JSON.
5. Add an entry in a `tasks.json` file:
```json
{
  "MyTask": {
    "task": "Extraction",
    "data": {
      "extracts": "path/to/extracts.jsonl",
      "schema": "path/to/property_schema.json",
      "ground_truth": "path/to/ground_truth_entities.jsonl",
      "metrics_config": "path/to/metrics_config.json"
    }
  }
}

* run evaluation (from this directory):

```bash
uv run run.py \
  --predictions ./data/predictions.jsonl \
  --output_dir ./output_dir
```

This will produce an output directory with:
- `metrics.json` – AESOP metrics - per file and aggregated across dataset
- `debug_output/` directory with debug .xlsx files if specified in `metrics_config.json`.