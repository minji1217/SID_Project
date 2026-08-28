from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pprint import pformat
from typing import Any

import polars as pl

from . import config
from .build_train_events import build_train_events
from .build_validation_events import build_validation_events


# =============================================================================
# Purpose
# =============================================================================
# This is an ablation experiment:
#
#   normalize_v2 canonical entity
#       + already-saved wikidata_candidates.parquet
#       -> deterministic GLOBAL canonical-key -> QID mapping
#       -> same Event builder (0.3 / 72h / 1%)
#
# No GPT call.
# No Wikidata API call.
#
# IMPORTANT:
# Unlike the GPT version, the mapping here is GLOBAL per canonical entity key.
# Therefore the same canonical key can never become WD::Q... in one article
# and fall back to TYPE::entity in another article.
# This lets us test whether GPT's article-by-article fragmentation was hurting
# Event clustering.
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize-v2 + cached Wikidata-only Event ablation"
    )
    parser.add_argument(
        "--policy",
        choices=["strict", "top1"],
        default="strict",
        help=(
            "strict: link only conservative unambiguous candidates (recommended). "
            "top1: always use the first Wikidata candidate unless it is an explicit TYPE_MISMATCH."
        ),
    )
    return parser.parse_args()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm_text(value: Any) -> str:
    """
    Conservative comparison normalization used only for comparing
    canonical surface vs Wikidata label. It does NOT alter saved entity keys.
    """
    text = _safe_text(value).casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _candidate_rows_by_key(candidate_df: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    required = {
        "original_canonical_entity_key",
        "canonical_entity",
        "candidate_index",
        "qid",
        "label",
        "type_diagnostic",
    }
    missing = required - set(candidate_df.columns)
    if missing:
        raise ValueError(
            "wikidata_candidates.parquet에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )

    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in candidate_df.iter_rows(named=True):
        key = _safe_text(row.get("original_canonical_entity_key"))
        qid = _safe_text(row.get("qid"))

        if not key or not qid.startswith("Q"):
            continue

        by_key[key].append(row)

    for key in by_key:
        # Deterministic order. Same QID may theoretically appear more than once;
        # de-duplicate while preserving the best candidate_index.
        best_by_qid: dict[str, dict[str, Any]] = {}
        for row in by_key[key]:
            qid = _safe_text(row["qid"])
            old = best_by_qid.get(qid)
            if old is None:
                best_by_qid[qid] = row
                continue

            old_idx = int(old.get("candidate_index") or 10**9)
            new_idx = int(row.get("candidate_index") or 10**9)
            if new_idx < old_idx:
                best_by_qid[qid] = row

        by_key[key] = sorted(
            best_by_qid.values(),
            key=lambda r: (
                int(r.get("candidate_index") or 10**9),
                _safe_text(r.get("qid")),
            ),
        )

    return dict(by_key)


def _choose_strict_candidate(
    canonical_key: str,
    rows: list[dict[str, Any]],
) -> tuple[str | None, str]:
    """
    Conservative Wikidata-only linking.

    Rule order:
      1) Exactly one distinct candidate and it is not TYPE_MISMATCH.
      2) Multiple candidates, but exactly one candidate is TYPE_MATCH.
      3) Multiple candidates, but exactly one candidate label exactly matches
         normalize_v2 canonical_entity (case-insensitive), and it is not TYPE_MISMATCH.
      4) Otherwise keep normalize_v2 key.

    This intentionally prefers precision over recall.
    """
    if not rows:
        return None, "NO_CANDIDATE"

    # 1) only one QID exists
    if len(rows) == 1:
        row = rows[0]
        if _safe_text(row.get("type_diagnostic")) != "TYPE_MISMATCH":
            return _safe_text(row["qid"]), "UNIQUE_CANDIDATE"
        return None, "UNIQUE_BUT_TYPE_MISMATCH"

    # 2) exactly one type match among all candidates
    type_matches = [
        r
        for r in rows
        if _safe_text(r.get("type_diagnostic")) == "TYPE_MATCH"
    ]
    if len(type_matches) == 1:
        return _safe_text(type_matches[0]["qid"]), "UNIQUE_TYPE_MATCH"

    # 3) exactly one exact label match
    canonical_surface = _norm_text(rows[0].get("canonical_entity"))
    if canonical_surface:
        exact_label_matches = [
            r
            for r in rows
            if _norm_text(r.get("label")) == canonical_surface
            and _safe_text(r.get("type_diagnostic")) != "TYPE_MISMATCH"
        ]
        if len(exact_label_matches) == 1:
            return _safe_text(exact_label_matches[0]["qid"]), "UNIQUE_EXACT_LABEL"

    return None, "AMBIGUOUS_FALLBACK"


def _choose_top1_candidate(
    canonical_key: str,
    rows: list[dict[str, Any]],
) -> tuple[str | None, str]:
    """
    Diagnostic policy only.
    Use candidate_index #1 globally unless it is an explicit TYPE_MISMATCH.
    """
    if not rows:
        return None, "NO_CANDIDATE"

    top = rows[0]

    if _safe_text(top.get("type_diagnostic")) == "TYPE_MISMATCH":
        return None, "TOP1_TYPE_MISMATCH_FALLBACK"

    return _safe_text(top["qid"]), "TOP1"


def _build_global_mapping(
    article_entities_df: pl.DataFrame,
    candidate_df: pl.DataFrame,
    policy: str,
) -> tuple[dict[str, str], pl.DataFrame, dict[str, Any]]:
    required = {
        "article_id",
        "canonical_entity_key",
    }
    missing = required - set(article_entities_df.columns)
    if missing:
        raise ValueError(
            "article_entities.parquet에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )

    # canonical_entity is not required from article_entities because
    # wikidata_candidates already carries it.
    all_keys = sorted(
        {
            _safe_text(x)
            for x in article_entities_df.get_column("canonical_entity_key").to_list()
            if _safe_text(x)
        }
    )

    rows_by_key = _candidate_rows_by_key(candidate_df)

    final_key_by_canonical: dict[str, str] = {}
    audit_rows: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()

    for canonical_key in all_keys:
        candidate_rows = rows_by_key.get(canonical_key, [])

        if policy == "strict":
            selected_qid, reason = _choose_strict_candidate(
                canonical_key,
                candidate_rows,
            )
        elif policy == "top1":
            selected_qid, reason = _choose_top1_candidate(
                canonical_key,
                candidate_rows,
            )
        else:
            raise ValueError(f"Unknown policy: {policy}")

        final_key = (
            f"WD::{selected_qid}"
            if selected_qid
            else canonical_key
        )

        final_key_by_canonical[canonical_key] = final_key
        reason_counter[reason] += 1

        selected_row = None
        if selected_qid:
            for r in candidate_rows:
                if _safe_text(r.get("qid")) == selected_qid:
                    selected_row = r
                    break

        audit_rows.append(
            {
                "original_canonical_entity_key": canonical_key,
                "selected_qid": selected_qid,
                "final_entity_key": final_key,
                "was_linked": selected_qid is not None,
                "mapping_reason": reason,
                "candidate_count": len(candidate_rows),
                "selected_candidate_index": (
                    int(selected_row.get("candidate_index"))
                    if selected_row is not None
                    and selected_row.get("candidate_index") is not None
                    else None
                ),
                "selected_label": (
                    _safe_text(selected_row.get("label"))
                    if selected_row is not None
                    else None
                ),
                "selected_type_diagnostic": (
                    _safe_text(selected_row.get("type_diagnostic"))
                    if selected_row is not None
                    else None
                ),
            }
        )

    mapping_df = pl.DataFrame(
        audit_rows,
        schema={
            "original_canonical_entity_key": pl.Utf8,
            "selected_qid": pl.Utf8,
            "final_entity_key": pl.Utf8,
            "was_linked": pl.Boolean,
            "mapping_reason": pl.Utf8,
            "candidate_count": pl.Int64,
            "selected_candidate_index": pl.Int64,
            "selected_label": pl.Utf8,
            "selected_type_diagnostic": pl.Utf8,
        },
    )

    linked_count = sum(
        1 for k, v in final_key_by_canonical.items() if k != v
    )

    mapping_summary = {
        "policy": policy,
        "canonical_entity_key_count": len(all_keys),
        "mapped_to_wikidata_qid_count": linked_count,
        "fallback_normalize_v2_count": len(all_keys) - linked_count,
        "mapping_reason_counts": dict(sorted(reason_counter.items())),
    }

    return final_key_by_canonical, mapping_df, mapping_summary


def _build_article_wikidata_only_entities(
    article_entities_df: pl.DataFrame,
    final_key_by_canonical: dict[str, str],
    selected_article_ids: list[int],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    by_article: dict[int, set[str]] = defaultdict(set)

    article_entity_case_count = 0
    changed_article_entity_case_count = 0

    for row in article_entities_df.iter_rows(named=True):
        article_id = row.get("article_id")
        canonical_key = _safe_text(row.get("canonical_entity_key"))

        if article_id is None or not canonical_key:
            continue

        article_id = int(article_id)
        final_key = final_key_by_canonical.get(
            canonical_key,
            canonical_key,
        )

        article_entity_case_count += 1
        if final_key != canonical_key:
            changed_article_entity_case_count += 1

        by_article[article_id].add(final_key)

    rows = [
        {
            "article_id": int(article_id),
            # Keep the same column name expected by linked_entity_input.py.
            "linked_entities": sorted(
                by_article.get(int(article_id), set())
            ),
        }
        for article_id in selected_article_ids
    ]

    article_df = pl.DataFrame(
        rows,
        schema={
            "article_id": pl.Int64,
            "linked_entities": pl.List(pl.Utf8),
        },
    )

    empty_count = sum(
        1
        for entities in article_df.get_column("linked_entities").to_list()
        if not entities
    )

    final_unique_keys = {
        entity
        for entities in article_df.get_column("linked_entities").to_list()
        for entity in (entities or [])
    }

    article_summary = {
        "selected_article_count": len(selected_article_ids),
        "article_entity_case_count": article_entity_case_count,
        "changed_article_entity_case_count": changed_article_entity_case_count,
        "empty_entity_article_count": empty_count,
        "final_unique_entity_key_count": len(final_unique_keys),
    }

    return article_df, article_summary


def _redirect_event_config(policy: str) -> dict[str, Any]:
    experiment_name = f"event_wikidata_only_{policy}"
    output_dir = config.EXPERIMENTS_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    wikidata_candidates_path = (
        config.EXPERIMENTS_DIR
        / "entity_linking_full"
        / "wikidata_candidates.parquet"
    )

    # Same input column contract used by the Entity-Linking Event runner.
    config.ARTICLE_LINKED_ENTITIES_PATH = (
        output_dir / "article_wikidata_only_entities.parquet"
    )

    # Separate outputs: never overwrite normalize_v2 or GPT-linked Event outputs.
    config.EVENT_OUTPUT_DIR = output_dir
    config.ARTICLE_EVENTS_PATH = output_dir / "article_events.parquet"
    config.EVENT_MASTER_PATH = output_dir / "event_master.parquet"
    config.ENTITY_IDF_PATH = output_dir / "entity_idf.parquet"
    config.VALIDATION_ARTICLE_EVENTS_PATH = (
        output_dir / "validation_article_events.parquet"
    )
    config.EVENT_MASTER_WITH_VALIDATION_PATH = (
        output_dir / "event_master_with_validation.parquet"
    )
    config.ALL_ARTICLE_EVENTS_PATH = (
        output_dir / "all_article_events.parquet"
    )
    config.EVENT_BUILD_SUMMARY_PATH = (
        output_dir / "event_build_summary.txt"
    )

    return {
        "output_dir": output_dir,
        "wikidata_candidates_path": wikidata_candidates_path,
        "mapping_path": output_dir / "wikidata_only_mapping.parquet",
    }


def _read_selected_article_ids() -> list[int]:
    train_df = pl.read_parquet(
        config.TRAIN_USED_ARTICLE_IDS_PATH,
        columns=["article_id"],
    )
    validation_df = pl.read_parquet(
        config.VALIDATION_ONLY_ARTICLE_IDS_PATH,
        columns=["article_id"],
    )

    ids = {
        int(x)
        for x in train_df.get_column("article_id").to_list()
        if x is not None
    }
    ids.update(
        int(x)
        for x in validation_df.get_column("article_id").to_list()
        if x is not None
    )

    return sorted(ids)


def _write_summary(
    policy: str,
    paths: dict[str, Any],
    mapping_summary: dict[str, Any],
    article_summary: dict[str, Any],
    train_result: dict[str, Any],
    validation_result: dict[str, Any],
) -> None:
    lines = [
        "SID_Project Normalize-v2 + Wikidata-only Event Summary",
        "=" * 70,
        f"policy={policy}",
        f"normalize_v2_input={config.FROZEN_NORMALIZE_V2_DIR}",
        f"wikidata_candidates={paths['wikidata_candidates_path']}",
        f"output_dir={paths['output_dir']}",
        f"similarity={config.EVENT_ENTITY_SIMILARITY_THRESHOLD}",
        f"time_window_hours={config.EVENT_TIME_WINDOW_HOURS}",
        f"max_entity_df_ratio={config.EVENT_MAX_ENTITY_DF_RATIO}",
        "openai_api_used=False",
        "wikidata_api_used=False",
        "global_surface_to_qid_policy=True",
        "",
        "[Mapping Result]",
        pformat(mapping_summary, sort_dicts=False),
        "",
        "[Article Entity Result]",
        pformat(article_summary, sort_dicts=False),
        "",
        "[Train Event Result]",
        pformat(train_result, sort_dicts=False),
        "",
        "[Validation Event Result]",
        pformat(validation_result, sort_dicts=False),
        "",
        "[Outputs]",
        f"mapping={paths['mapping_path']}",
        f"article_wikidata_only_entities={config.ARTICLE_LINKED_ENTITIES_PATH}",
        f"article_events={config.ARTICLE_EVENTS_PATH}",
        f"event_master={config.EVENT_MASTER_PATH}",
        f"entity_idf={config.ENTITY_IDF_PATH}",
        f"validation_article_events={config.VALIDATION_ARTICLE_EVENTS_PATH}",
        f"event_master_with_validation={config.EVENT_MASTER_WITH_VALIDATION_PATH}",
        f"all_article_events={config.ALL_ARTICLE_EVENTS_PATH}",
    ]

    config.EVENT_BUILD_SUMMARY_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    policy = args.policy

    paths = _redirect_event_config(policy)

    article_entities_path = (
        config.FROZEN_NORMALIZE_V2_DIR
        / "article_entities.parquet"
    )
    candidate_path = paths["wikidata_candidates_path"]

    required_paths = [
        article_entities_path,
        candidate_path,
        config.TRAIN_USED_ARTICLE_IDS_PATH,
        config.VALIDATION_ONLY_ARTICLE_IDS_PATH,
    ]

    missing = [p for p in required_paths if not p.exists()]
    if missing:
        text = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            "필요한 입력 파일이 없습니다:\n" + text
        )

    print("=" * 80)
    print("SID_Project Normalize-v2 + Wikidata-only Event experiment")
    print("=" * 80)
    print(f"Policy           : {policy}")
    print(f"Normalize v2     : {article_entities_path}")
    print(f"Wikidata cached  : {candidate_path}")
    print(f"New output       : {paths['output_dir']}")
    print(
        "Frozen params    : "
        f"similarity={config.EVENT_ENTITY_SIMILARITY_THRESHOLD}, "
        f"time_window={config.EVENT_TIME_WINDOW_HOURS}h, "
        f"max_entity_df_ratio={config.EVENT_MAX_ENTITY_DF_RATIO}"
    )
    print("OpenAI API       : NOT USED")
    print("Wikidata API     : NOT USED")
    print()

    print("[STEP 0] Building GLOBAL normalize_v2 -> Wikidata mapping...")
    article_entities_df = pl.read_parquet(article_entities_path)
    candidate_df = pl.read_parquet(candidate_path)

    (
        final_key_by_canonical,
        mapping_df,
        mapping_summary,
    ) = _build_global_mapping(
        article_entities_df=article_entities_df,
        candidate_df=candidate_df,
        policy=policy,
    )

    mapping_df.write_parquet(
        paths["mapping_path"],
        compression="zstd",
    )

    print("[OK] Global Wikidata mapping completed.")
    print(pformat(mapping_summary, sort_dicts=False))
    print()

    print("[STEP 1] Building article Wikidata-only entity sets...")
    selected_article_ids = _read_selected_article_ids()

    article_df, article_summary = _build_article_wikidata_only_entities(
        article_entities_df=article_entities_df,
        final_key_by_canonical=final_key_by_canonical,
        selected_article_ids=selected_article_ids,
    )

    article_df.write_parquet(
        config.ARTICLE_LINKED_ENTITIES_PATH,
        compression="zstd",
    )

    print("[OK] Article entity input completed.")
    print(pformat(article_summary, sort_dicts=False))
    print()

    print("[STEP 2] Building Train Events with SAME Event engine...")
    train_result = build_train_events()
    print("[OK] Train Event build completed.")
    print(pformat(train_result, sort_dicts=False))
    print()

    print("[STEP 3] Building Validation Events with SAME Event engine...")
    validation_result = build_validation_events()
    print("[OK] Validation Event build completed.")
    print(pformat(validation_result, sort_dicts=False))
    print()

    _write_summary(
        policy=policy,
        paths=paths,
        mapping_summary=mapping_summary,
        article_summary=article_summary,
        train_result=train_result,
        validation_result=validation_result,
    )

    print("=" * 80)
    print("COMPLETED")
    print(f"Outputs: {paths['output_dir']}")
    print(
        "normalize_v2, entity_linking_full, and event_entity_linking "
        "were NOT modified."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
