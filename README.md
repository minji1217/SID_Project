# EB-NeRD event-aware Semantic ID

This repository currently implements **STEP 0 only**: read-only inspection and validation of raw EB-NeRD Parquet inputs. Article preprocessing, embeddings, event clustering, RQ-VAE, and Transformer training are intentionally out of scope.

## Setup and run

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m preprocess.run_validation
pytest
```

Input and output locations are project-relative and configured in `configs/paths.yaml`; required schemas and report thresholds are in `configs/preprocessing.yaml`.

The validation runner reads `data/articles.parquet`, `data/train/`, and `data/validation/` without changing them. It writes JSON reports to `outputs/reports/` and missing-ID audit Parquets to `outputs/audit/`.

## Checks

- Schema, null, list-length, and required-column inspection for all five raw inputs.
- Article ID, history alignment/order/repeat, and behavior click-list/order validation.
- Cross-file article referential integrity plus train/validation user and article overlap statistics.
- Stable click deduplication is used for statistics only; no raw or processed input is mutated in STEP 0.
