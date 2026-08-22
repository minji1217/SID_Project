# SID_Project Entity-Linking Event Experiment

This package rebuilds Events using the completed GPT/Wikidata Entity Linking result while preserving the existing `src/` and frozen `normalize_v2` artifacts.

## Input

- `data/output/experiments/entity_linking_full/article_linked_entities.parquet`
- frozen upstream metadata from `data/output/experiments/normalize_v2/model_inputs/`

## Frozen Event policy

- `EVENT_ENTITY_SIMILARITY_THRESHOLD = 0.3`
- `EVENT_TIME_WINDOW_HOURS = 72`
- `EVENT_MAX_ENTITY_DF_RATIO = 0.01`

Only the Entity representation changes.

## Run

From the `SID_Project` root:

```powershell
python -m event_entity_linking.run_event_build
```

Train only:

```powershell
python -m event_entity_linking.run_event_build --train-only
```

Validation only (requires the new Train Event outputs first):

```powershell
python -m event_entity_linking.run_event_build --validation-only
```

## Output

`data/output/experiments/event_entity_linking/`

- `article_events.parquet`
- `event_master.parquet`
- `entity_idf.parquet`
- `validation_article_events.parquet`
- `event_master_with_validation.parquet`
- `all_article_events.parquet`
- `event_build_summary.txt`

No OpenAI or Wikidata API is called in this package.
