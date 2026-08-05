# Repository Guidelines

## Project Structure & Scope

Build the EB-NeRD event-aware Semantic ID pipeline: validate and preprocess data, embed articles, cluster fixed events, train an event-aware RQ-VAE, then train an encoder-decoder Transformer to predict the next clicked article SID `(c1, c2, c3)`.

- `data/` holds immutable raw `articles.parquet`, `train/`, and `validation/` inputs.
- `check.yaml` records dataset column mappings.
- `a.py` is an early inspection/conversion utility.
- Put reusable code under `src/`, tests under `tests/`, and generated data, audits, and reports under `output/` or `processed/`.

Never modify raw Parquet files or implement a different recommendation task without approval.

## Development Commands

Activate the local environment and run scripts from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python a.py
pytest
```

Use Python 3.10+, Polars for dataframe work, and PyArrow only for unsupported Parquet operations. Add dependencies to the project’s environment before relying on them.

## Coding & Data Rules

Use four-space Python indentation, `snake_case`, `pathlib.Path`, type hints on public functions, YAML configuration, and structured logging. Keep schema names and thresholds configurable rather than hard-coded.

History list fields are positionally aligned: validate lengths, sort all paired fields together when required, retain repeat visits, and report corrections. For behavior rows, stable-deduplicate clicks, retain exactly-one-click targets, create samples before appending the target, and audit excluded rows.

Build text from usable title/subtitle only; use L2-normalized float32 768-dimensional `intfloat/multilingual-e5-base` embeddings. Fit categories and event clusters on train data only; assign validation articles inductively. Preserve original NER mentions and never silently merge aliases.

## Testing & Validation

Write `pytest` tests as `tests/test_<module>.py` using small synthetic fixtures. Every pipeline stage must validate schemas, log before/after row counts, emit a JSON report, retain audit files, and fail fast on structural corruption.

## Commits & Pull Requests

Use concise imperative commits, e.g. `Validate history alignment`. Keep changes focused. In pull requests, state affected data/stages, assumptions, tests run, new outputs or dependencies, and link an issue when available. Include sample reports or screenshots for generated artifacts.
