# EB-NeRD Event-aware Semantic ID Project

## 1. Project goal

Build an end-to-end pipeline:

1. Validate and preprocess EB-NeRD data.
2. Generate article embeddings.
3. Build fixed event clusters.
4. Train an event-aware RQ-VAE.
5. Generate article Semantic IDs `(c1, c2, c3)`.
6. Train an encoder-decoder Transformer that generates the next clicked article SID.

Do not implement a different recommendation problem without explicit approval.

---

## 2. Language and runtime

- Python 3.10 or later
- Use Polars as the primary dataframe library.
- Use PyArrow only when Polars cannot handle a required Parquet operation.
- Use pathlib instead of manually concatenating paths.
- Use type hints for public functions.
- Use YAML configuration instead of hard-coded paths and thresholds.
- Use logging instead of uncontrolled print statements.
- Use pytest for unit tests.

---

## 3. Raw files

Expected raw inputs:

- `articles.parquet`
- `train/behaviors.parquet`
- `train/history.parquet`
- `validation/behaviors.parquet`
- `validation/history.parquet`

Never modify raw files.

All generated files must be written under an output or processed directory.

---

## 4. History rules

Each history row belongs to one user.

The following list columns are positionally aligned:

- `article_id_fixed[i]`
- `impression_time_fixed[i]`
- `read_time_fixed[i]`
- `scroll_percentage_fixed[i]`

Required validation:

- Check list-length equality for every user.
- Check whether timestamps are already ascending for every user.
- Do not sort article IDs independently.
- When a user is unsorted, zip all aligned fields, sort them together by timestamp, and report that the row was reordered.
- Preserve repeated article visits.
- Do not globally deduplicate `article_id_fixed`.
- Report consecutive duplicate article IDs separately.

---

## 5. Behavior rules

Use these columns:

- `impression_id`
- `user_id`
- `impression_time`
- `article_id`
- `article_ids_clicked`

Do not use `article_ids_inview` as the Transformer baseline input.

Click processing:

1. Stable-deduplicate `article_ids_clicked`.
2. Keep rows whose unique click count is exactly one.
3. Set that article as `target_article_id`.
4. Save multi-click and duplicate-click rows to audit files.

Sequence processing:

1. Start from the user's `article_id_fixed`.
2. Sort behaviors by exact `impression_time`.
3. If `article_id` is non-null and differs from the last history article, append it.
4. Generate the sample before adding the target.
5. Append `target_article_id` after the sample is generated.
6. Avoid only consecutive duplicates.
7. Never put the current target into its own input history.

---

## 6. Article text and category

Build:

`model_text = "query: " + title + "\n" + subtitle`

Rules:

- Strip whitespace.
- Do not emit literal `"None"` or `"null"`.
- Exclude articles with no usable title and subtitle.
- Use `intfloat/multilingual-e5-base`.
- Output dimension: 768.
- L2-normalize embeddings.
- Save embeddings as float32.

Build `model_category_id` only from `category_str`.

- Fit mapping using train articles only.
- Reserve `0` for `<UNK>`.
- Entity linking is unrelated to `model_category_id`.

---

## 7. NER rules

Create `entity_key_unlinked` from:

`entity_type + "::" + normalized mention`

Example:

`PER::zlatan ibrahimovic`

Rules:

- Preserve every original mention in the mention table.
- Deduplicate only when producing an article-level entity set for event clustering.
- Do not assume `PER::chris` and `PER::chris hytn` are the same person.
- Entity linking is a separate experiment.
- Do not silently merge aliases.

---

## 8. Event clustering rules

- Fit event clusters using train articles only.
- Do not jointly cluster train and validation articles.
- Validation-only articles must be assigned inductively after train events are fixed.
- `event_id` is an offline cluster identifier.
- `event_id` is not the RQ-VAE c2 code.
- Store event membership and event statistics.

---

## 9. RQ-VAE rules

Input fields:

- `article_id`
- `embedding_row`
- `model_category_id`
- `event_id`
- `published_at`

Architecture:

- Input embedding: 768
- Encoder: 768 → 256 → 128
- Decoder: 128 → 256 → 768
- Semantic ID: `(c1, c2, c3)`

Training:

- Use random article-level mini-batches.
- Do not create event-grouped batches.
- c1 and c3 are article-level.
- c2 is looked up by `event_id` from a global event cache.
- Refresh the event cache at epoch start.
- Hold the event cache fixed during the epoch.
- Do not claim that the cached event representation receives exact end-to-end gradients.

---

## 10. Transformer rules

Architecture:

- Encoder-decoder Transformer
- Not decoder-only

Task:

`past article SID sequence → next clicked article SID`

Encoder input:

- Flatten article SIDs in history order.
- Use disjoint token spaces for C1, C2, and C3.
- PAD, BOS, and EOS must not overlap with SID tokens.

Decoder:

- Input: `[BOS, C1, C2, C3]`
- Labels: `[C1, C2, C3, EOS]`

Do not create candidate click labels such as 0 or 1.

---

## 11. Validation policy

Every pipeline stage must:

1. Validate required schemas.
2. Produce a machine-readable JSON report.
3. Log row counts before and after transformation.
4. Report dropped rows and reasons.
5. Fail fast on structural corruption.
6. Preserve audit files for policy-based exclusions.
7. Include deterministic tests using small synthetic fixtures.

Do not silently fix data without reporting the correction.

---

## 12. Change policy

Before modifying code:

1. Inspect the existing repository.
2. Summarize the files that will change.
3. State assumptions.
4. Do not modify unrelated files.

After modifying code:

1. Run relevant tests.
2. Run formatting or static checks when configured.
3. Summarize modified files.
4. Report commands executed.
5. Report remaining limitations honestly.