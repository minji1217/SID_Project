from __future__ import annotations

"""
SID_Project - full Entity Linking standalone runner.

Frozen policy implemented here:
    Safe Normalization v2 (read-only)
      -> Wikidata candidate generation (da + en)
      -> candidate P31/P279 metadata (TYPE is soft evidence only)
      -> article context + same-article entities + GPT disambiguation
      -> LINKED / AMBIGUOUS / UNLINKED

Important safeguards:
- Never writes into baseline/ or normalize_v2/.
- Entity Linking is decided per (article_id, canonical_entity_key), not globally per surface.
- GPT may select ONLY a supplied candidate QID; Python validates this strictly.
- AMBIGUOUS / UNLINKED keeps the normalize_v2 canonical key unchanged.
- No-candidate cases skip GPT entirely and become UNLINKED.
- Wikidata search/entity metadata and GPT responses are persistent JSONL caches.
- Re-running resumes from caches.
- --limit-articles 20 uses the exact same code path as the full run.

This script intentionally does NOT rebuild Events. After linking is reviewed, the existing
Event algorithm should consume article_linked_entities.parquet through a minimal connector,
while keeping similarity=0.3, time_window=72h, high-DF=1% unchanged.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import smtplib
import traceback
import time
import threading
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import polars as pl
import requests
from openai import OpenAI
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Fixed experiment policy / versions
# -----------------------------------------------------------------------------

PROMPT_VERSION = "sid_entity_linking_lean_v2_20260821"
WIKIDATA_SEARCH_VERSION = "sid_wikidata_search_v1_20260818"
WIKIDATA_ENTITY_VERSION = "sid_wikidata_entity_meta_v1_20260818"

DEFAULT_MODEL = "gpt-5-mini-2025-08-07"  # pinned snapshot for reproducibility

# Estimated direct OpenAI API text-token pricing for the pinned model.
# Verified against official OpenAI pricing on 2026-08-21.
# USD per 1,000,000 tokens.
MODEL_TOKEN_PRICING_USD_PER_1M = {
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-mini-2025-08-07": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
}

DEFAULT_INPUT_DIR = Path("data/output/experiments/normalize_v2/model_inputs")
DEFAULT_OUTPUT_DIR = Path("data/output/experiments/entity_linking_full")

DEFAULT_WIKIDATA_API = "https://www.wikidata.org/w/api.php"
DEFAULT_WIKIDATA_LANGS = ("da", "en")

DECISION_LINKED = "LINKED"
DECISION_AMBIGUOUS = "AMBIGUOUS"
DECISION_UNLINKED = "UNLINKED"

# Event settings are shown here only as a guardrail/reminder. This script does NOT alter
# or run the Event clustering algorithm.
EVENT_ENTITY_SIMILARITY_THRESHOLD = 0.3
EVENT_TIME_WINDOW_HOURS = 72
EVENT_MAX_ENTITY_DF_RATIO = 0.01



# -----------------------------------------------------------------------------
# Optional email progress notifier (standard library SMTP)
# -----------------------------------------------------------------------------

class EmailNotifier:
    """Best-effort SMTP notifier. Email failures never stop Entity Linking."""

    def __init__(
        self,
        *,
        recipient: str,
        username: str,
        app_password: str,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
    ):
        self.recipient = recipient
        self.username = username
        self.app_password = app_password
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    def send(self, subject: str, body: str) -> bool:
        msg = EmailMessage()
        msg["From"] = self.username
        msg["To"] = self.recipient
        msg["Subject"] = subject
        msg.set_content(body)

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(self.username, self.app_password)
                smtp.send_message(msg)
            info(f"Email notification sent: {subject}")
            return True
        except Exception as exc:
            warn(
                f"Email notification failed ({type(exc).__name__}: {exc}). "
                "Entity Linking will continue."
            )
            return False


def build_email_notifier(args: argparse.Namespace) -> Optional[EmailNotifier]:
    recipient = safe_text(args.notify_email or os.getenv("NOTIFY_EMAIL"))
    if not recipient:
        return None

    username = safe_text(os.getenv("SMTP_USERNAME"))
    app_password = safe_text(os.getenv("SMTP_APP_PASSWORD")).replace(" ", "")

    if not username or not app_password:
        warn(
            "--notify-email was set, but SMTP_USERNAME / SMTP_APP_PASSWORD "
            "environment variables are missing. Email notifications are disabled."
        )
        return None

    return EmailNotifier(
        recipient=recipient,
        username=username,
        app_password=app_password,
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
    )


_EMAIL_NOTIFIER_FOR_FATAL_ERROR: Optional[EmailNotifier] = None

# -----------------------------------------------------------------------------
# Structured GPT output
# -----------------------------------------------------------------------------

DecisionName = Literal["LINKED", "AMBIGUOUS", "UNLINKED"]
ConfidenceName = Literal["HIGH", "MEDIUM", "LOW"]


class GPTEntityDecision(BaseModel):
    entity_key: str = Field(description="Exact entity_key supplied in the request")
    decision: DecisionName
    selected_qid: str = Field(
        description="For LINKED: one supplied candidate QID. For AMBIGUOUS/UNLINKED: empty string."
    )
    confidence: ConfidenceName
    reason: str = Field(description="Very short evidence-based reason, ideally <= 12 words, grounded in supplied evidence")


class GPTBatchDecision(BaseModel):
    decisions: List[GPTEntityDecision]


SYSTEM_PROMPT = """You are an entity-linking disambiguator for a research pipeline.

For each TARGET entity supplied in `entities`, choose exactly one:
- LINKED: exactly one supplied Wikidata candidate clearly matches the real entity in this article.
- AMBIGUOUS: two or more supplied candidates remain plausible.
- UNLINKED: none of the supplied candidates is sufficiently supported.

Hard rules:
1. Output decisions ONLY for entity_key values present in `entities`.
2. `same_article_entities` are CONTEXT ONLY. Never output decisions for them.
3. For LINKED, selected_qid MUST be one of that target's supplied candidate QIDs.
4. Never invent, repair, infer, or output any other QID.
5. NER type, Wikidata type labels, type diagnostics, and search rank are SOFT evidence only.
6. Exact surface match or search rank 1 is not enough. Prioritize article title/context and co-entities.
7. If the correct real-world entity is absent from supplied candidates, choose UNLINKED.
8. If multiple candidates remain plausible, choose AMBIGUOUS rather than guessing.
9. Return exactly one decision for every target entity_key, using the exact supplied entity_key.
10. For AMBIGUOUS or UNLINKED, selected_qid must be an empty string.
11. Keep reason extremely short: one brief sentence, ideally 12 words or fewer.

Do not provide chain-of-thought or lengthy explanations.
"""


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()


def normalize_surface_cache_key(surface: str) -> str:
    return " ".join(surface.strip().casefold().split())


def chunks(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def chunk_objects(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = safe_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"


def parse_key(entity_key: str) -> Tuple[str, str]:
    if "::" not in entity_key:
        return "UNK", entity_key
    group, surface = entity_key.split("::", 1)
    return group, surface


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr, flush=True)


def info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


class JsonlCache:
    """Append-only JSONL cache. Latest record for a cache_key wins."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = record.get("cache_key")
                    if key:
                        self.records[str(key)] = record
                except json.JSONDecodeError:
                    # A process may have been killed during the final append. Skip only the
                    # malformed line; prior completed cache lines remain usable.
                    warn(f"Skipping malformed JSONL cache line: {self.path}:{lineno}")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.records.get(key)

    def put(self, record: Dict[str, Any]) -> None:
        key = str(record["cache_key"])
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.records[key] = record

    def __len__(self) -> int:
        with self._lock:
            return len(self.records)


# -----------------------------------------------------------------------------
# Input schema resolution
# -----------------------------------------------------------------------------


def resolve_column(
    columns: Sequence[str],
    candidates: Sequence[str],
    *,
    required: bool = True,
    override: Optional[str] = None,
    logical_name: str = "column",
) -> Optional[str]:
    if override:
        if override not in columns:
            raise ValueError(f"Requested {logical_name} override '{override}' not found. columns={list(columns)}")
        return override
    lower_map = {c.casefold(): c for c in columns}
    for candidate in candidates:
        actual = lower_map.get(candidate.casefold())
        if actual:
            return actual
    if required:
        raise ValueError(
            f"Could not resolve {logical_name}. Tried {list(candidates)}. Actual columns={list(columns)}"
        )
    return None


@dataclass
class ResolvedSchema:
    entity_article_id: str
    entity_group: Optional[str]
    canonical_entity: Optional[str]
    canonical_key: Optional[str]
    article_article_id: str
    title: str
    context_columns: List[str]


def resolve_schema(entity_df: pl.DataFrame, article_df: pl.DataFrame, args: argparse.Namespace) -> ResolvedSchema:
    ecols = entity_df.columns
    acols = article_df.columns

    entity_article_id = resolve_column(
        ecols, ["article_id", "articleId", "id"], override=args.entity_article_id_column, logical_name="entity article_id"
    )
    entity_group = resolve_column(
        ecols,
        ["entity_group", "raw_entity_group", "ner_type", "entity_type", "type"],
        required=False,
        override=args.entity_group_column,
        logical_name="entity_group",
    )
    canonical_key = resolve_column(
        ecols,
        ["canonical_entity_key", "original_canonical_entity_key", "entity_key", "normalized_entity_key", "baseline_entity_key"],
        required=False,
        override=args.canonical_key_column,
        logical_name="canonical_entity_key",
    )
    canonical_entity = resolve_column(
        ecols,
        ["canonical_entity", "normalized_entity", "entity", "surface", "normalized_surface", "raw_entity"],
        required=False,
        override=args.canonical_entity_column,
        logical_name="canonical_entity",
    )
    if canonical_key is None and (entity_group is None or canonical_entity is None):
        raise ValueError(
            "Need either a canonical key column, or both entity_group and canonical_entity columns. "
            f"Entity columns={ecols}"
        )

    article_article_id = resolve_column(
        acols, ["article_id", "articleId", "id"], override=args.article_article_id_column, logical_name="article article_id"
    )
    title = resolve_column(
        acols, ["title", "article_title", "headline"], override=args.title_column, logical_name="title"
    )

    if args.context_columns:
        context_columns = []
        for c in args.context_columns:
            if c not in acols:
                raise ValueError(f"Context column '{c}' not found. Article columns={acols}")
            context_columns.append(c)
    else:
        # Optional text context. We intentionally do not require any of these.
        preferred = [
            "body",
            "content",
            "article_text",
            "text",
            "description",
            "subtitle",
            "summary",
            "lead",
        ]
        context_columns = []
        lower_map = {c.casefold(): c for c in acols}
        for name in preferred:
            actual = lower_map.get(name.casefold())
            if actual and actual != title and actual not in context_columns:
                context_columns.append(actual)
        # Keep context compact and deterministic.
        context_columns = context_columns[:2]

    return ResolvedSchema(
        entity_article_id=entity_article_id,
        entity_group=entity_group,
        canonical_entity=canonical_entity,
        canonical_key=canonical_key,
        article_article_id=article_article_id,
        title=title,
        context_columns=context_columns,
    )


def read_scope_ids(path: Path, preferred_column: Optional[str] = None) -> List[Tuple[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required scope file not found: {path}")
    df = pl.read_parquet(path)
    id_col = resolve_column(
        df.columns,
        [preferred_column] if preferred_column else ["article_id", "articleId", "id"],
        required=True,
        logical_name=f"article_id in {path.name}",
    )
    rows: List[Tuple[str, Any]] = []
    seen: set[str] = set()
    for value in df.get_column(id_col).to_list():
        if value is None:
            continue
        key = str(value)
        if key not in seen:
            seen.add(key)
            rows.append((key, value))
    return rows


def choose_scope(
    train_rows: List[Tuple[str, Any]],
    validation_rows: List[Tuple[str, Any]],
    limit_articles: Optional[int],
) -> Tuple[List[Tuple[str, Any, str]], Dict[str, bool]]:
    train_map = {k: v for k, v in train_rows}
    validation_map = {k: v for k, v in validation_rows if k not in train_map}

    overlap = len(validation_rows) - len(validation_map)
    if overlap:
        warn(f"Found {overlap} article IDs in both train and validation scope; train takes priority.")

    train_sorted = sorted(train_map.items(), key=lambda kv: kv[0])
    val_sorted = sorted(validation_map.items(), key=lambda kv: kv[0])

    selected: List[Tuple[str, Any, str]] = []
    if limit_articles is None:
        selected.extend((k, v, "TRAIN") for k, v in train_sorted)
        selected.extend((k, v, "VALIDATION") for k, v in val_sorted)
    else:
        if limit_articles <= 0:
            raise ValueError("--limit-articles must be > 0")
        # Technical smoke test: deliberately exercise both Train and Validation when possible.
        train_n = min(len(train_sorted), (limit_articles + 1) // 2)
        val_n = min(len(val_sorted), limit_articles - train_n)
        if train_n + val_n < limit_articles:
            extra = limit_articles - (train_n + val_n)
            train_n = min(len(train_sorted), train_n + extra)
        if train_n + val_n < limit_articles:
            extra = limit_articles - (train_n + val_n)
            val_n = min(len(val_sorted), val_n + extra)
        selected.extend((k, v, "TRAIN") for k, v in train_sorted[:train_n])
        selected.extend((k, v, "VALIDATION") for k, v in val_sorted[:val_n])

    is_train = {k: split == "TRAIN" for k, _v, split in selected}
    return selected, is_train


def build_normalized_entity_pairs(
    entity_df: pl.DataFrame,
    schema: ResolvedSchema,
    scope_keys: set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    cols = [schema.entity_article_id]
    for c in [schema.entity_group, schema.canonical_entity, schema.canonical_key]:
        if c and c not in cols:
            cols.append(c)

    rows: List[Dict[str, Any]] = []
    by_article: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_pairs: set[Tuple[str, str]] = set()

    for raw in entity_df.select(cols).iter_rows(named=True):
        article_value = raw.get(schema.entity_article_id)
        if article_value is None:
            continue
        article_key = str(article_value)
        if article_key not in scope_keys:
            continue

        canonical_key = safe_text(raw.get(schema.canonical_key)) if schema.canonical_key else ""
        entity_group = safe_text(raw.get(schema.entity_group)) if schema.entity_group else ""
        canonical_entity = safe_text(raw.get(schema.canonical_entity)) if schema.canonical_entity else ""

        if canonical_key:
            key_group, key_surface = parse_key(canonical_key)
            if not entity_group:
                entity_group = key_group
            if not canonical_entity:
                canonical_entity = key_surface
        else:
            if not entity_group or not canonical_entity:
                continue
            canonical_key = f"{entity_group}::{canonical_entity}"

        pair_key = (article_key, canonical_key)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        row = {
            "article_id_key": article_key,
            "entity_group": entity_group or "UNK",
            "canonical_entity": canonical_entity,
            "original_canonical_entity_key": canonical_key,
        }
        rows.append(row)
        by_article[article_key].append(row)

    for article_key in by_article:
        by_article[article_key].sort(key=lambda r: r["original_canonical_entity_key"])
    rows.sort(key=lambda r: (r["article_id_key"], r["original_canonical_entity_key"]))
    return rows, by_article


def build_article_context_map(
    article_df: pl.DataFrame,
    schema: ResolvedSchema,
    scope_keys: set[str],
    context_chars: int,
) -> Dict[str, Dict[str, str]]:
    cols = [schema.article_article_id, schema.title] + schema.context_columns
    result: Dict[str, Dict[str, str]] = {}
    for row in article_df.select(cols).iter_rows(named=True):
        article_value = row.get(schema.article_article_id)
        if article_value is None:
            continue
        article_key = str(article_value)
        if article_key not in scope_keys or article_key in result:
            continue
        title = safe_text(row.get(schema.title))
        context_parts = [safe_text(row.get(c)) for c in schema.context_columns]
        context_parts = [x for x in context_parts if x]
        context = truncate_text("\n\n".join(context_parts), context_chars)
        result[article_key] = {"title": title, "context": context}
    return result


# -----------------------------------------------------------------------------
# Wikidata client + persistent caches
# -----------------------------------------------------------------------------


class WikidataClient:
    def __init__(
        self,
        *,
        api_url: str,
        user_agent: str,
        search_cache: JsonlCache,
        entity_cache: JsonlCache,
        langs: Sequence[str],
        limit_per_lang: int,
        max_candidates: int,
        timeout_s: float,
        delay_s: float,
        max_retries: int,
    ):
        self.api_url = api_url
        self.search_cache = search_cache
        self.entity_cache = entity_cache
        self.langs = tuple(langs)
        self.limit_per_lang = limit_per_lang
        self.max_candidates = max_candidates
        self.timeout_s = timeout_s
        self.delay_s = delay_s
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self.stats = Counter()

    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        params.setdefault("format", "json")
        params.setdefault("maxlag", 5)
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            if self.delay_s > 0:
                time.sleep(self.delay_s)
            try:
                response = self.session.get(self.api_url, params=params, timeout=self.timeout_s)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 2 ** attempt)
                    warn(f"Wikidata 429; respecting retry delay ({sleep_s}s).")
                    time.sleep(sleep_s)
                    continue
                response.raise_for_status()
                data = response.json()
                if "error" in data:
                    code = safe_text(data["error"].get("code"))
                    if code == "maxlag":
                        time.sleep(min(60.0, 2 ** attempt))
                        continue
                    raise RuntimeError(f"Wikidata API error: {data['error']}")
                return data
            except Exception as exc:  # network / HTTP / JSON
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                sleep_s = min(60.0, (2 ** attempt) + random.random())
                warn(f"Wikidata request failed ({type(exc).__name__}); retrying after backoff.")
                time.sleep(sleep_s)

        raise RuntimeError(f"Wikidata request failed after {self.max_retries} attempts: {last_error}")

    def search_cache_key(self, surface: str) -> str:
        return sha256_json(
            {
                "version": WIKIDATA_SEARCH_VERSION,
                "surface": normalize_surface_cache_key(surface),
                "langs": self.langs,
                "limit_per_lang": self.limit_per_lang,
                "max_candidates": self.max_candidates,
            }
        )

    def search_surface(self, surface: str, *, allow_network: bool = True) -> Dict[str, Any]:
        cache_key = self.search_cache_key(surface)
        cached = self.search_cache.get(cache_key)
        if cached is not None:
            self.stats["search_cache_hit"] += 1
            return cached
        if not allow_network:
            raise RuntimeError(
                f"--link-only requires a complete Wikidata search cache, but surface is missing: {surface!r}"
            )

        combined: Dict[str, Dict[str, Any]] = {}
        for lang in self.langs:
            data = self._get(
                {
                    "action": "wbsearchentities",
                    "search": surface,
                    "language": lang,
                    "uselang": lang,
                    "type": "item",
                    "limit": self.limit_per_lang,
                }
            )
            self.stats["search_http_request"] += 1
            for rank, item in enumerate(data.get("search", []), 1):
                qid = safe_text(item.get("id"))
                if not qid.startswith("Q"):
                    continue
                hit = {
                    "language": lang,
                    "rank": rank,
                    "label": safe_text(item.get("label")),
                    "description": safe_text(item.get("description")),
                    "match_type": safe_text((item.get("match") or {}).get("type")),
                    "match_text": safe_text((item.get("match") or {}).get("text")),
                }
                if qid not in combined:
                    combined[qid] = {"qid": qid, "search_hits": []}
                combined[qid]["search_hits"].append(hit)

        def candidate_sort_key(c: Dict[str, Any]) -> Tuple[int, int, str]:
            hits = c["search_hits"]
            best_rank = min((int(h["rank"]) for h in hits), default=999)
            best_lang_index = min(
                (self.langs.index(h["language"]) for h in hits if h["language"] in self.langs),
                default=999,
            )
            return best_rank, best_lang_index, c["qid"]

        candidates = sorted(combined.values(), key=candidate_sort_key)[: self.max_candidates]
        record = {
            "cache_key": cache_key,
            "version": WIKIDATA_SEARCH_VERSION,
            "surface": surface,
            "normalized_surface": normalize_surface_cache_key(surface),
            "langs": list(self.langs),
            "limit_per_lang": self.limit_per_lang,
            "max_candidates": self.max_candidates,
            "candidates": candidates,
        }
        self.search_cache.put(record)
        self.stats["search_cache_write"] += 1
        return record

    @staticmethod
    def _claim_item_qids(entity: Dict[str, Any], prop: str) -> List[str]:
        result: List[str] = []
        for claim in (entity.get("claims") or {}).get(prop, []):
            try:
                value = claim["mainsnak"]["datavalue"]["value"]
                qid = safe_text(value.get("id"))
                if qid.startswith("Q") and qid not in result:
                    result.append(qid)
            except Exception:
                continue
        return result

    def entity_cache_key(self, qid: str) -> str:
        return sha256_json({"version": WIKIDATA_ENTITY_VERSION, "qid": qid, "langs": self.langs})

    def get_entity_meta(self, qid: str) -> Optional[Dict[str, Any]]:
        record = self.entity_cache.get(self.entity_cache_key(qid))
        return record.get("entity") if record else None

    def ensure_entity_meta(self, qids: Sequence[str], *, allow_network: bool = True) -> None:
        unique_qids = sorted({q for q in qids if q and q.startswith("Q")})
        missing = [q for q in unique_qids if self.get_entity_meta(q) is None]
        if not missing:
            self.stats["entity_meta_cache_hit"] += len(unique_qids)
            return
        self.stats["entity_meta_cache_hit"] += len(unique_qids) - len(missing)
        if not allow_network:
            preview = ", ".join(missing[:10])
            raise RuntimeError(
                f"--link-only requires complete Wikidata entity metadata cache; missing {len(missing)} QIDs: {preview}"
            )

        for batch in chunks(missing, 50):
            data = self._get(
                {
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": "labels|descriptions|claims",
                    "languages": "|".join(self.langs),
                    "languagefallback": 1,
                }
            )
            self.stats["entity_meta_http_request"] += 1
            entities = data.get("entities") or {}
            for qid in batch:
                entity = entities.get(qid) or {}
                labels = {
                    lang: safe_text((entity.get("labels") or {}).get(lang, {}).get("value"))
                    for lang in self.langs
                }
                descriptions = {
                    lang: safe_text((entity.get("descriptions") or {}).get(lang, {}).get("value"))
                    for lang in self.langs
                }
                parsed = {
                    "qid": qid,
                    "missing": bool(entity.get("missing") is not None),
                    "labels": labels,
                    "descriptions": descriptions,
                    "p31_qids": self._claim_item_qids(entity, "P31"),
                    "p279_qids": self._claim_item_qids(entity, "P279"),
                }
                self.entity_cache.put(
                    {
                        "cache_key": self.entity_cache_key(qid),
                        "version": WIKIDATA_ENTITY_VERSION,
                        "entity": parsed,
                    }
                )
                self.stats["entity_meta_cache_write"] += 1

    def enrich_candidate_type_metadata(self, candidate_qids: Sequence[str], *, allow_network: bool = True) -> None:
        # Layer 0: candidate items themselves.
        self.ensure_entity_meta(candidate_qids, allow_network=allow_network)

        # Layer 1: P31 / direct P279 classes.
        class_qids: set[str] = set()
        for qid in candidate_qids:
            meta = self.get_entity_meta(qid) or {}
            class_qids.update(meta.get("p31_qids") or [])
            class_qids.update(meta.get("p279_qids") or [])
        self.ensure_entity_meta(sorted(class_qids), allow_network=allow_network)

        # Layer 2: one P279 level above those classes. This is diagnostic context only,
        # never a hard filter.
        parent_qids: set[str] = set()
        for qid in class_qids:
            meta = self.get_entity_meta(qid) or {}
            parent_qids.update(meta.get("p279_qids") or [])
        self.ensure_entity_meta(sorted(parent_qids), allow_network=allow_network)

    def best_label(self, qid: str) -> str:
        meta = self.get_entity_meta(qid) or {}
        labels = meta.get("labels") or {}
        for lang in self.langs:
            label = safe_text(labels.get(lang))
            if label:
                return label
        return qid

    def best_description(self, qid: str) -> str:
        meta = self.get_entity_meta(qid) or {}
        descriptions = meta.get("descriptions") or {}
        for lang in self.langs:
            description = safe_text(descriptions.get(lang))
            if description:
                return description
        return ""

    def type_payload(self, qid: str) -> Dict[str, Any]:
        meta = self.get_entity_meta(qid) or {}
        instance_of: List[Dict[str, Any]] = []
        for class_qid in meta.get("p31_qids") or []:
            class_meta = self.get_entity_meta(class_qid) or {}
            parent_types = [
                {"qid": parent_qid, "label": self.best_label(parent_qid)}
                for parent_qid in (class_meta.get("p279_qids") or [])[:8]
            ]
            instance_of.append(
                {
                    "qid": class_qid,
                    "label": self.best_label(class_qid),
                    "parent_types": parent_types,
                }
            )
        direct_subclass_of = [
            {"qid": parent_qid, "label": self.best_label(parent_qid)}
            for parent_qid in (meta.get("p279_qids") or [])[:8]
        ]
        return {"instance_of": instance_of[:8], "direct_subclass_of": direct_subclass_of}

    def soft_type_diagnostic(self, ner_type: str, qid: str) -> str:
        """A deliberately conservative diagnostic. It NEVER controls candidate retention."""
        ner = safe_text(ner_type).upper()
        payload = self.type_payload(qid)
        instance_qids = {x["qid"] for x in payload["instance_of"]}
        labels = []
        for x in payload["instance_of"]:
            labels.append(x["label"].casefold())
            labels.extend(p["label"].casefold() for p in x["parent_types"])
        labels.extend(x["label"].casefold() for x in payload["direct_subclass_of"])
        label_text = " | ".join(labels)

        if not payload["instance_of"] and not payload["direct_subclass_of"]:
            return "TYPE_UNKNOWN"
        if ner == "PER":
            return "TYPE_MATCH" if "Q5" in instance_qids else "TYPE_MISMATCH"
        if ner == "ORG":
            org_words = (
                "organization",
                "organisation",
                "company",
                "business",
                "club",
                "team",
                "association",
                "agency",
                "institution",
                "government",
                "corporation",
                "enterprise",
                "political party",
            )
            return "TYPE_MATCH" if any(w in label_text for w in org_words) else "TYPE_UNKNOWN"
        if ner == "LOC":
            loc_words = (
                "geographic",
                "geographical",
                "city",
                "country",
                "municipality",
                "village",
                "town",
                "district",
                "region",
                "island",
                "river",
                "lake",
                "sea",
                "territorial",
                "location",
                "place",
            )
            return "TYPE_MATCH" if any(w in label_text for w in loc_words) else "TYPE_UNKNOWN"
        return "TYPE_UNCHECKED"

    def candidate_payload(self, ner_type: str, search_candidate: Dict[str, Any]) -> Dict[str, Any]:
        qid = search_candidate["qid"]
        type_payload = self.type_payload(qid)
        return {
            "qid": qid,
            "label": self.best_label(qid),
            "description": self.best_description(qid),
            "search_hits": search_candidate.get("search_hits") or [],
            "type_diagnostic": self.soft_type_diagnostic(ner_type, qid),
            "instance_of": type_payload["instance_of"],
            "direct_subclass_of": type_payload["direct_subclass_of"],
        }

    @staticmethod
    def _unique_nonempty(values: Iterable[str], limit: int) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for value in values:
            value = safe_text(value)
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
            if len(out) >= limit:
                break
        return out

    def candidate_payload_for_gpt(
        self,
        ner_type: str,
        search_candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compact GPT-only candidate representation.

        This intentionally preserves candidate identity and useful semantic evidence while
        removing repeated/nested serialization overhead:
        - KEEP candidate QID, label, description.
        - KEEP per-language best search rank and matched text (compact).
        - KEEP the existing soft type diagnostic.
        - KEEP P31/P279 semantic labels, but flatten them into short unique label arrays.
        - DROP repeated search-hit labels/descriptions and P31/P279 nested QID dictionaries.

        The original candidate_payload() remains unchanged for audit parquet output.
        """
        qid = search_candidate["qid"]
        hits = search_candidate.get("search_hits") or []
        type_payload = self.type_payload(qid)

        search_ranks: Dict[str, int] = {}
        matched_texts: List[str] = []
        for hit in hits:
            lang = safe_text(hit.get("language"))
            rank_raw = hit.get("rank")
            if lang and rank_raw is not None:
                try:
                    rank = int(rank_raw)
                    if lang not in search_ranks or rank < search_ranks[lang]:
                        search_ranks[lang] = rank
                except (TypeError, ValueError):
                    pass

            match_text = safe_text(hit.get("match_text"))
            if match_text:
                matched_texts.append(match_text)

        instance_labels = self._unique_nonempty(
            (item.get("label", "") for item in type_payload["instance_of"]),
            limit=6,
        )

        parent_label_values: List[str] = []
        for item in type_payload["instance_of"]:
            parent_label_values.extend(
                safe_text(parent.get("label"))
                for parent in (item.get("parent_types") or [])
            )
        parent_label_values.extend(
            safe_text(item.get("label"))
            for item in type_payload["direct_subclass_of"]
        )
        parent_labels = self._unique_nonempty(parent_label_values, limit=8)

        return {
            "qid": qid,
            "label": self.best_label(qid),
            "description": self.best_description(qid),
            "search_ranks": search_ranks,
            "matched_texts": self._unique_nonempty(matched_texts, limit=2),
            "type_diagnostic": self.soft_type_diagnostic(ner_type, qid),
            "instance_of_labels": instance_labels,
            "parent_type_labels": parent_labels,
        }


# -----------------------------------------------------------------------------
# OpenAI GPT linker + validation + cache
# -----------------------------------------------------------------------------


class GPTLinker:
    def __init__(
        self,
        *,
        client: OpenAI,
        cache: JsonlCache,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
        max_retries: int,
    ):
        self.client = client
        self.cache = cache
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.stats = Counter()
        self._stats_lock = threading.Lock()

    def _add_stat(self, key: str, value: int = 1) -> None:
        with self._stats_lock:
            self.stats[key] += int(value)

    def stats_snapshot(self) -> Counter:
        with self._stats_lock:
            return Counter(self.stats)

    @staticmethod
    def semantic_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Payload actually hashed/sent to GPT; private pipeline bookkeeping is excluded."""
        return {k: v for k, v in payload.items() if not str(k).startswith("_")}

    def cache_key(self, payload: Dict[str, Any]) -> str:
        return sha256_json(
            {
                "prompt_version": PROMPT_VERSION,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "payload": self.semantic_payload(payload),
            }
        )

    @staticmethod
    def _usage_to_dict(response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)
        cached = 0
        details = getattr(usage, "input_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        return {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def validate_decision_subset(payload: Dict[str, Any], parsed: GPTBatchDecision) -> List[Dict[str, Any]]:
        """Validate every returned decision, while allowing a proper subset of requested keys.

        This is used only to recover from a *complete-structure but incomplete-cardinality*
        model response (for example: 2 entities requested but only 1 decision returned).
        Unknown keys, duplicate keys, invalid QIDs, and non-empty QIDs for unresolved
        decisions are still rejected immediately.
        """
        supplied_entities = payload["entities"]
        supplied_by_key = {e["entity_key"]: e for e in supplied_entities}
        expected_keys = set(supplied_by_key)

        decisions = [d.model_dump() if hasattr(d, "model_dump") else d.dict() for d in parsed.decisions]
        returned_keys = [safe_text(d.get("entity_key")) for d in decisions]

        if len(set(returned_keys)) != len(returned_keys):
            raise ValueError(f"GPT returned duplicate entity_key values: {returned_keys}")

        unknown_keys = set(returned_keys) - expected_keys
        if unknown_keys:
            raise ValueError(
                f"GPT returned entity_key values that were not supplied: {sorted(unknown_keys)}; "
                f"expected subset of {sorted(expected_keys)}"
            )

        for decision in decisions:
            entity_key = decision["entity_key"]
            status = decision["decision"]
            selected_qid = safe_text(decision.get("selected_qid"))
            allowed_qids = {c["qid"] for c in supplied_by_key[entity_key]["candidates"]}

            if status == DECISION_LINKED:
                if selected_qid not in allowed_qids:
                    # Critical hallucination/constraint guardrail.
                    raise ValueError(
                        f"GPT selected non-supplied QID for {entity_key}: {selected_qid!r}; allowed={sorted(allowed_qids)}"
                    )
            else:
                if selected_qid:
                    raise ValueError(
                        f"{status} must have empty selected_qid for {entity_key}, got {selected_qid!r}"
                    )

        decisions.sort(key=lambda d: d["entity_key"])
        return decisions

    @classmethod
    def validate_decisions(cls, payload: Dict[str, Any], parsed: GPTBatchDecision) -> List[Dict[str, Any]]:
        """Strict final validation: exactly one valid decision for every requested entity."""
        supplied_entities = payload["entities"]
        expected_keys = {e["entity_key"] for e in supplied_entities}
        decisions = cls.validate_decision_subset(payload, parsed)
        returned_keys = [safe_text(d.get("entity_key")) for d in decisions]

        if len(returned_keys) != len(expected_keys):
            missing = sorted(expected_keys - set(returned_keys))
            raise ValueError(
                f"GPT returned {len(returned_keys)} decisions; expected {len(expected_keys)}; missing={missing}"
            )
        if set(returned_keys) != expected_keys:
            raise ValueError(
                f"GPT entity_key set mismatch. expected={sorted(expected_keys)}, returned={sorted(returned_keys)}"
            )
        return decisions

    @staticmethod
    def _merge_usage(*usage_dicts: Dict[str, int]) -> Dict[str, int]:
        keys = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
        return {key: sum(int(u.get(key, 0) or 0) for u in usage_dicts) for key in keys}

    @staticmethod
    def _payload_for_keys(
        payload: Dict[str, Any],
        entity_keys: set[str],
        *,
        repair_issues: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a repair payload with identical article/context but only selected entities.

        A repair instruction is embedded in the user payload so the cache key is distinct from
        the original batch and the model is explicitly reminded to stay inside supplied QIDs.
        """
        repair_payload = dict(payload)
        repair_payload["entities"] = [
            e for e in payload["entities"] if e["entity_key"] in entity_keys
        ]
        issue_map = repair_issues or {}
        allowed_by_key = {
            e["entity_key"]: [c["qid"] for c in e.get("candidates") or []]
            for e in repair_payload["entities"]
        }
        repair_payload["repair_instruction"] = {
            "reason": "Previous output for these entities was missing or invalid and must be repaired.",
            "hard_rule": (
                "For LINKED, selected_qid MUST be one of the supplied allowed_qids for that exact entity_key. "
                "If the real-world entity is not among those candidates, return UNLINKED with empty selected_qid. "
                "Do not invent or use any other Wikidata QID."
            ),
            "issues": {k: issue_map.get(k, "missing decision") for k in sorted(entity_keys)},
            "allowed_qids": {k: allowed_by_key.get(k, []) for k in sorted(entity_keys)},
        }
        return repair_payload

    @staticmethod
    def _partition_returned_decisions(
        payload: Dict[str, Any], parsed: GPTBatchDecision
    ) -> Tuple[List[Dict[str, Any]], set[str], Dict[str, str], List[str]]:
        """Split a model response into valid decisions and entity keys that need repair.

        This function never accepts an invalid QID. It only salvages decisions that are already
        fully valid, and marks missing/duplicate/invalid decisions for a focused repair request.
        Unknown entity keys are ignored and reported because they cannot be mapped to a supplied
        entity safely.
        """
        supplied_entities = payload["entities"]
        supplied_by_key = {e["entity_key"]: e for e in supplied_entities}
        expected_keys = set(supplied_by_key)

        decisions = [d.model_dump() if hasattr(d, "model_dump") else d.dict() for d in parsed.decisions]
        by_key: Dict[str, List[Dict[str, Any]]] = {}
        unknown_keys: List[str] = []
        for decision in decisions:
            key = safe_text(decision.get("entity_key"))
            if key not in expected_keys:
                unknown_keys.append(key)
                continue
            by_key.setdefault(key, []).append(decision)

        valid: List[Dict[str, Any]] = []
        repair_keys: set[str] = set()
        issues: Dict[str, str] = {}

        for key in sorted(expected_keys):
            rows = by_key.get(key, [])
            if not rows:
                repair_keys.add(key)
                issues[key] = "missing decision"
                continue
            if len(rows) != 1:
                repair_keys.add(key)
                issues[key] = f"duplicate decisions returned ({len(rows)})"
                continue

            decision = rows[0]
            status = decision["decision"]
            selected_qid = safe_text(decision.get("selected_qid"))
            allowed_qids = {c["qid"] for c in supplied_by_key[key].get("candidates") or []}

            if status == DECISION_LINKED:
                if selected_qid not in allowed_qids:
                    repair_keys.add(key)
                    issues[key] = (
                        f"selected non-supplied QID {selected_qid!r}; "
                        f"allowed={sorted(allowed_qids)}"
                    )
                    continue
            elif selected_qid:
                repair_keys.add(key)
                issues[key] = f"{status} returned non-empty selected_qid {selected_qid!r}"
                continue

            valid.append(decision)

        valid.sort(key=lambda d: d["entity_key"])
        return valid, repair_keys, issues, unknown_keys

    @staticmethod
    def _fallback_unlinked_decisions(
        entity_keys: set[str],
        repair_issues: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Conservative deterministic fallback after one focused repair failure.

        We NEVER accept a non-supplied QID.  If GPT still cannot produce a valid
        decision after the focused repair attempt, preserve the original canonical
        TYPE::entity key by emitting UNLINKED/LOW for only those affected targets.
        """
        issues = repair_issues or {}
        out: List[Dict[str, Any]] = []
        for key in sorted(entity_keys):
            issue = safe_text(issues.get(key))
            if "non-supplied QID" in issue:
                short_reason = "REPAIR_FALLBACK_UNLINKED: non-supplied QID repeated"
            elif "missing decision" in issue:
                short_reason = "REPAIR_FALLBACK_UNLINKED: decision repeatedly missing"
            elif "duplicate" in issue:
                short_reason = "REPAIR_FALLBACK_UNLINKED: duplicate decisions repeated"
            else:
                short_reason = "REPAIR_FALLBACK_UNLINKED: invalid repair response"
            out.append(
                {
                    "entity_key": key,
                    "decision": DECISION_UNLINKED,
                    "selected_qid": "",
                    "confidence": "LOW",
                    "reason": short_reason,
                }
            )
        return out

    def link(self, payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        cache_key = self.cache_key(payload)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._add_stat("gpt_cache_hit")
            decisions = cached.get("decisions") or []
            # Validate cached data too, so a stale/corrupt cache can never inject a QID.
            parsed = GPTBatchDecision(decisions=[GPTEntityDecision(**d) for d in decisions])
            validated = self.validate_decisions(payload, parsed)
            return validated, cached

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(self.semantic_payload(payload), ensure_ascii=False, separators=(",", ":"))},
                    ],
                    text_format=GPTBatchDecision,
                    reasoning={"effort": self.reasoning_effort},
                    max_output_tokens=self.max_output_tokens,
                    store=False,
                )

                # Count usage as soon as an API response exists. This includes
                # responses later rejected by validation and retried.
                usage = self._usage_to_dict(response)
                self._add_stat("api_response_count")
                self._add_stat("new_input_tokens", usage["input_tokens"])
                self._add_stat("new_cached_input_tokens", usage["cached_input_tokens"])
                self._add_stat("new_output_tokens", usage["output_tokens"])

                parsed = response.output_parsed
                if parsed is None:
                    raise ValueError("OpenAI response.output_parsed is None (refusal/incomplete/unparseable response)")

                # Salvage every decision that is already valid, and repair only entity keys
                # whose decision is missing, duplicated, or semantically invalid (for example
                # a LINKED decision selecting a QID that was not supplied).  No invalid QID is
                # ever accepted or converted silently.
                partial_decisions, repair_keys, repair_issues, unknown_keys = (
                    self._partition_returned_decisions(payload, parsed)
                )
                expected_keys = {e["entity_key"] for e in payload["entities"]}
                repair_cache_keys: List[str] = []
                fallback_unlinked_keys: List[str] = []
                combined_usage = usage

                if unknown_keys:
                    warn(
                        f"GPT returned non-supplied entity_key value(s) {sorted(set(unknown_keys))}; "
                        "ignoring them and repairing only affected supplied entities."
                    )

                if repair_keys:
                    # Runtime repair policy (2026-08-21):
                    #
                    # ORIGINAL request:
                    #   valid targets are preserved; only missing/invalid targets are
                    #   sent once to a focused repair request.
                    #
                    # FOCUSED repair request:
                    #   if GPT still returns any missing/invalid target decision,
                    #   DO NOT spend 4 more validation retries on the same semantic
                    #   mistake.  Conservatively emit UNLINKED/LOW for only those
                    #   affected targets.  This preserves TYPE::entity downstream and
                    #   never accepts a hallucinated/non-supplied QID.
                    is_focused_repair = "repair_instruction" in payload
                    all_require_repair = (
                        not partial_decisions and repair_keys == expected_keys
                    )

                    missing_count = sum(
                        1
                        for k in repair_keys
                        if repair_issues.get(k) == "missing decision"
                    )
                    invalid_count = len(repair_keys) - missing_count

                    if is_focused_repair:
                        warn(
                            f"Focused repair still returned {len(repair_keys)} invalid/missing "
                            f"decision(s); using conservative UNLINKED fallback instead of "
                            f"retrying the same repair 5 times "
                            f"(missing={missing_count}, invalid={invalid_count}): "
                            f"{sorted(repair_keys)}"
                        )
                        fallback_decisions = self._fallback_unlinked_decisions(
                            repair_keys, repair_issues
                        )
                        fallback_unlinked_keys.extend(sorted(repair_keys))

                        combined_decisions = partial_decisions + fallback_decisions
                        combined_parsed = GPTBatchDecision(
                            decisions=[
                                GPTEntityDecision(**d) for d in combined_decisions
                            ]
                        )
                        decisions = self.validate_decisions(
                            payload, combined_parsed
                        )

                    else:
                        if all_require_repair:
                            warn(
                                f"All {len(repair_keys)} returned decision(s) require repair; "
                                f"switching to one focused repair request "
                                f"(missing={missing_count}, invalid={invalid_count}): "
                                f"{sorted(repair_keys)}"
                            )
                        else:
                            warn(
                                f"Repairing only {len(repair_keys)} entity decision(s) "
                                f"(missing={missing_count}, invalid={invalid_count}): "
                                f"{sorted(repair_keys)}"
                            )

                        repair_payload = self._payload_for_keys(
                            payload, repair_keys, repair_issues=repair_issues
                        )
                        try:
                            repair_decisions, repair_record = self.link(
                                repair_payload
                            )
                            repair_cache_keys.append(
                                repair_record["cache_key"]
                            )
                            fallback_unlinked_keys.extend(
                                repair_record.get("fallback_unlinked_keys") or []
                            )

                            combined_decisions = (
                                partial_decisions + repair_decisions
                            )
                            combined_parsed = GPTBatchDecision(
                                decisions=[
                                    GPTEntityDecision(**d)
                                    for d in combined_decisions
                                ]
                            )
                            decisions = self.validate_decisions(
                                payload, combined_parsed
                            )
                            combined_usage = self._merge_usage(
                                usage, repair_record.get("usage") or {}
                            )
                        except RuntimeError as repair_exc:
                            # Network/refusal/parse failures may still exhaust the
                            # focused repair's technical retry loop.  Do not crash the
                            # entire 12,860-article run: fall back only affected keys.
                            warn(
                                f"Focused repair exhausted technical retries "
                                f"({type(repair_exc).__name__}: {repair_exc}); "
                                f"using conservative UNLINKED fallback for "
                                f"{sorted(repair_keys)}"
                            )
                            fallback_decisions = (
                                self._fallback_unlinked_decisions(
                                    repair_keys, repair_issues
                                )
                            )
                            fallback_unlinked_keys.extend(
                                sorted(repair_keys)
                            )
                            combined_decisions = (
                                partial_decisions + fallback_decisions
                            )
                            combined_parsed = GPTBatchDecision(
                                decisions=[
                                    GPTEntityDecision(**d)
                                    for d in combined_decisions
                                ]
                            )
                            decisions = self.validate_decisions(
                                payload, combined_parsed
                            )
                else:
                    # _partition_returned_decisions() has already discarded unknown
                    # entity_key values and verified every expected target decision.
                    # Validate the CLEANED decisions rather than the raw model response,
                    # otherwise harmless extra keys would trigger a full API retry.
                    cleaned_parsed = GPTBatchDecision(
                        decisions=[
                            GPTEntityDecision(**d) for d in partial_decisions
                        ]
                    )
                    decisions = self.validate_decisions(payload, cleaned_parsed)

                record = {
                    "cache_key": cache_key,
                    "prompt_version": PROMPT_VERSION,
                    "model": self.model,
                    "reasoning_effort": self.reasoning_effort,
                    "article_id": safe_text(payload.get("_article_id")),
                    "entity_keys": [e["entity_key"] for e in payload["entities"]],
                    "decisions": decisions,
                    "usage": combined_usage,
                    "response_id": safe_text(getattr(response, "id", "")),
                    "repair_cache_keys": repair_cache_keys,
                    "fallback_unlinked_keys": sorted(set(fallback_unlinked_keys)),
                }
                self.cache.put(record)

                # One cache write here corresponds to the current API response.  Any repair
                # request updates its own stats/cache inside the recursive self.link() call.
                self._add_stat("gpt_cache_write")
                return decisions, record
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                sleep_s = min(60.0, (2 ** attempt) + random.random())
                warn(
                    f"OpenAI call/output validation failed ({type(exc).__name__}: {exc}); retrying. "
                    "No invalid QID will be accepted."
                )
                time.sleep(sleep_s)

        # Research-safe behavior: do not silently convert a technical/API failure into a semantic UNLINKED.
        # All previous successful calls are cached, so the user can rerun and resume.
        raise RuntimeError(
            f"OpenAI linking failed after {self.max_retries} attempts. "
            f"Progress is preserved in GPT cache; rerun to resume. Last error: {last_error}"
        )


# -----------------------------------------------------------------------------
# Candidate/output builders
# -----------------------------------------------------------------------------


def collect_unique_surfaces(pair_rows: Sequence[Dict[str, Any]]) -> List[str]:
    surfaces = sorted({safe_text(r["canonical_entity"]) for r in pair_rows if safe_text(r["canonical_entity"])})
    return surfaces


def collect_all_candidate_qids(search_records: Dict[str, Dict[str, Any]]) -> List[str]:
    qids: set[str] = set()
    for record in search_records.values():
        for c in record.get("candidates") or []:
            qid = safe_text(c.get("qid"))
            if qid.startswith("Q"):
                qids.add(qid)
    return sorted(qids)


def build_wikidata_candidates_rows(
    unique_entity_keys: Sequence[Tuple[str, str, str]],
    surface_search_records: Dict[str, Dict[str, Any]],
    wd: WikidataClient,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entity_group, canonical_entity, canonical_key in sorted(unique_entity_keys, key=lambda x: x[2]):
        record = surface_search_records[normalize_surface_cache_key(canonical_entity)]
        candidates = record.get("candidates") or []
        for candidate_index, search_candidate in enumerate(candidates, 1):
            payload = wd.candidate_payload(entity_group, search_candidate)
            rows.append(
                {
                    "entity_group": entity_group,
                    "canonical_entity": canonical_entity,
                    "original_canonical_entity_key": canonical_key,
                    "candidate_index": candidate_index,
                    "qid": payload["qid"],
                    "label": payload["label"],
                    "description": payload["description"],
                    "type_diagnostic": payload["type_diagnostic"],
                    "search_hits_json": json.dumps(payload["search_hits"], ensure_ascii=False, sort_keys=True),
                    "instance_of_json": json.dumps(payload["instance_of"], ensure_ascii=False, sort_keys=True),
                    "direct_subclass_of_json": json.dumps(
                        payload["direct_subclass_of"], ensure_ascii=False, sort_keys=True
                    ),
                }
            )
    return rows


def build_gpt_payload(
    *,
    article_id_key: str,
    split: str,
    article_context: Dict[str, str],
    article_entities: Sequence[Dict[str, Any]],
    entity_chunk: Sequence[Dict[str, Any]],
    surface_search_records: Dict[str, Dict[str, Any]],
    wd: WikidataClient,
    max_coentities: int,
) -> Dict[str, Any]:
    # TARGET entities already appear below with surface/type/candidates.
    # Do not duplicate them inside same_article_entities.
    target_keys = {
        entity["original_canonical_entity_key"] for entity in entity_chunk
    }

    coentities: List[Dict[str, Any]] = []
    for entity in article_entities:
        if entity["original_canonical_entity_key"] in target_keys:
            continue
        coentities.append(
            {
                "surface": entity["canonical_entity"],
                "ner_type": entity["entity_group"],
            }
        )
        if len(coentities) >= max_coentities:
            break

    entities_payload: List[Dict[str, Any]] = []
    for entity in entity_chunk:
        search_record = surface_search_records[
            normalize_surface_cache_key(entity["canonical_entity"])
        ]
        candidates = [
            wd.candidate_payload_for_gpt(entity["entity_group"], candidate)
            for candidate in (search_record.get("candidates") or [])
        ]
        entities_payload.append(
            {
                "entity_key": entity["original_canonical_entity_key"],
                "surface": entity["canonical_entity"],
                "ner_type": entity["entity_group"],
                "candidates": candidates,
            }
        )

    # article_id_key and split are intentionally not sent to GPT: they are pipeline
    # bookkeeping, not semantic evidence. They remain available to the caller/cache record.
    return {
        "_article_id": article_id_key,
        "_split": split,
        "article": {
            "title": article_context.get("title", ""),
            "context": article_context.get("context", ""),
            "same_article_entities": coentities,
        },
        "entities": entities_payload,
    }


def build_article_entity_links(
    *,
    selected_scope: Sequence[Tuple[str, Any, str]],
    pairs_by_article: Dict[str, List[Dict[str, Any]]],
    article_context_map: Dict[str, Dict[str, str]],
    surface_search_records: Dict[str, Dict[str, Any]],
    wd: WikidataClient,
    gpt_linker: GPTLinker,
    gpt_entities_per_request: int,
    max_coentities: int,
    gpt_workers: int = 1,
    email_notifier: Optional[EmailNotifier] = None,
    notify_every_articles: int = 1000,
) -> List[Dict[str, Any]]:
    """Article-level Entity Linking with optional bounded parallel GPT calls.

    gpt_workers=1 preserves sequential behavior.
    gpt_workers>1 parallelizes only gpt_linker.link(payload).
    Payload building and final row assembly remain on the main thread.
    """
    scope_value = {k: v for k, v, _split in selected_scope}
    scope_split = {k: split for k, _v, split in selected_scope}
    link_rows: List[Dict[str, Any]] = []

    # Snapshot used only for progress logging.  "interval" means since the
    # previous progress log (normally ~100 submitted articles).  In parallel
    # mode a few requests may still be in flight, so this is intentionally
    # described as "since last progress" rather than an exact per-article bucket.
    last_progress_stats = Counter()

    def format_token_progress(stats: Counter) -> str:
        nonlocal last_progress_stats

        delta_cache_writes = stats["gpt_cache_write"] - last_progress_stats["gpt_cache_write"]
        delta_api_responses = stats["api_response_count"] - last_progress_stats["api_response_count"]
        delta_input = stats["new_input_tokens"] - last_progress_stats["new_input_tokens"]
        delta_cached = stats["new_cached_input_tokens"] - last_progress_stats["new_cached_input_tokens"]
        delta_output = stats["new_output_tokens"] - last_progress_stats["new_output_tokens"]

        # cached_input_tokens is a subset of input_tokens.
        delta_uncached = max(0, delta_input - delta_cached)
        delta_total = delta_input + delta_output

        cumulative_input = stats["new_input_tokens"]
        cumulative_cached = stats["new_cached_input_tokens"]
        cumulative_uncached = max(0, cumulative_input - cumulative_cached)
        cumulative_output = stats["new_output_tokens"]
        cumulative_total = cumulative_input + cumulative_output
        cumulative_api_responses = stats["api_response_count"]

        avg_input = delta_input / delta_api_responses if delta_api_responses > 0 else 0.0
        avg_output = delta_output / delta_api_responses if delta_api_responses > 0 else 0.0

        pricing = MODEL_TOKEN_PRICING_USD_PER_1M.get(gpt_linker.model)
        if pricing is None:
            cost_text = (
                f"estimated_cost=UNKNOWN(model={gpt_linker.model}); "
                "no pricing entry for this model"
            )
        else:
            interval_uncached_cost = delta_uncached / 1_000_000 * pricing["input"]
            interval_cached_cost = delta_cached / 1_000_000 * pricing["cached_input"]
            interval_output_cost = delta_output / 1_000_000 * pricing["output"]
            interval_cost = (
                interval_uncached_cost + interval_cached_cost + interval_output_cost
            )

            cumulative_uncached_cost = cumulative_uncached / 1_000_000 * pricing["input"]
            cumulative_cached_cost = cumulative_cached / 1_000_000 * pricing["cached_input"]
            cumulative_output_cost = cumulative_output / 1_000_000 * pricing["output"]
            cumulative_cost = (
                cumulative_uncached_cost
                + cumulative_cached_cost
                + cumulative_output_cost
            )

            cost_text = (
                f"estimated_cost_since_last=${interval_cost:.4f} "
                f"(uncached_input=${interval_uncached_cost:.4f}, "
                f"cached_input=${interval_cached_cost:.4f}, "
                f"output=${interval_output_cost:.4f}); "
                f"estimated_run_cost=${cumulative_cost:.4f}"
            )

        result = (
            f"tokens since last progress: total={delta_total:,} "
            f"(input={delta_input:,}, cached_input={delta_cached:,}, "
            f"uncached_input={delta_uncached:,}, output={delta_output:,}); "
            f"api_responses={delta_api_responses:,}, "
            f"successful_cache_writes={delta_cache_writes:,}, "
            f"avg_input/api_response={avg_input:,.0f}, "
            f"avg_output/api_response={avg_output:,.0f}; "
            f"{cost_text}; "
            f"run_cumulative_tokens={cumulative_total:,} "
            f"(input={cumulative_input:,}, cached_input={cumulative_cached:,}, "
            f"output={cumulative_output:,}, api_responses={cumulative_api_responses:,})"
        )

        last_progress_stats = Counter(stats)
        return result

    def maybe_email_progress(
        index: int,
        *,
        completed_requests: Optional[int] = None,
        submitted_requests: Optional[int] = None,
        in_flight: Optional[int] = None,
    ) -> None:
        if email_notifier is None or index % notify_every_articles != 0:
            return

        stats = gpt_linker.stats_snapshot()
        estimated_cost = estimate_gpt5_mini_cost(stats, gpt_linker.model)

        lines = [
            "SID_Project Entity Linking progress",
            "",
            f"articles = {index:,}/{len(selected_scope):,}",
            f"progress = {index / len(selected_scope) * 100:.1f}%",
            f"workers = {gpt_workers}",
            f"new GPT cache writes this run = {stats['gpt_cache_write']:,}",
            f"GPT cache hits this run = {stats['gpt_cache_hit']:,}",
            f"API responses this run = {stats['api_response_count']:,}",
            f"input tokens this run = {stats['new_input_tokens']:,}",
            f"output tokens this run = {stats['new_output_tokens']:,}",
        ]
        if estimated_cost is not None:
            lines.append(f"estimated API cost this run = ${estimated_cost:.4f}")
        if completed_requests is not None and submitted_requests is not None:
            lines += [
                f"GPT requests completed/submitted = "
                f"{completed_requests:,}/{submitted_requests:,}",
                f"in_flight = {in_flight or 0}",
            ]
        lines += [
            "",
            "The process is still running.",
        ]

        email_notifier.send(
            f"[SID Entity Linking] {index:,}/{len(selected_scope):,} articles",
            "\n".join(lines),
        )


    def prepare_article(
        article_key: str, split: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        article_entities = pairs_by_article.get(article_key, [])
        if not article_entities:
            return [], []

        gpt_needed: List[Dict[str, Any]] = []
        for entity in article_entities:
            search_record = surface_search_records[
                normalize_surface_cache_key(entity["canonical_entity"])
            ]
            candidate_count = len(search_record.get("candidates") or [])
            if candidate_count == 0:
                link_rows.append(
                    {
                        "article_id": scope_value[article_key],
                        "article_id_key": article_key,
                        "is_train_used": split == "TRAIN",
                        "split": split,
                        "entity_group": entity["entity_group"],
                        "canonical_entity": entity["canonical_entity"],
                        "original_canonical_entity_key": entity["original_canonical_entity_key"],
                        "decision": DECISION_UNLINKED,
                        "selected_qid": None,
                        "confidence": "LOW",
                        "reason": "NO_WIKIDATA_CANDIDATE",
                        "candidate_count": 0,
                        "linked_entity_key": entity["original_canonical_entity_key"],
                        "resolution_source": "NO_CANDIDATE",
                    }
                )
            else:
                gpt_needed.append(entity)
        return article_entities, gpt_needed

    def append_gpt_result_rows(
        article_key: str,
        split: str,
        entity_chunk: Sequence[Dict[str, Any]],
        decisions: Sequence[Dict[str, Any]],
        cache_record: Dict[str, Any],
    ) -> None:
        decision_by_key = {d["entity_key"]: d for d in decisions}

        for entity in entity_chunk:
            entity_key = entity["original_canonical_entity_key"]
            decision = decision_by_key[entity_key]
            selected_qid = safe_text(decision.get("selected_qid")) or None
            status = decision["decision"]

            search_record = surface_search_records[
                normalize_surface_cache_key(entity["canonical_entity"])
            ]
            allowed_qids = {c["qid"] for c in (search_record.get("candidates") or [])}

            if status == DECISION_LINKED:
                if selected_qid not in allowed_qids:
                    raise AssertionError(
                        f"Internal guardrail failure: selected QID {selected_qid} "
                        f"is not supplied for {entity_key}"
                    )
                linked_entity_key = f"WD::{selected_qid}"
            else:
                selected_qid = None
                linked_entity_key = entity_key

            is_repair_fallback = (
                entity_key
                in set(cache_record.get("fallback_unlinked_keys") or [])
                or safe_text(decision.get("reason")).startswith(
                    "REPAIR_FALLBACK_UNLINKED:"
                )
            )

            link_rows.append(
                {
                    "article_id": scope_value[article_key],
                    "article_id_key": article_key,
                    "is_train_used": split == "TRAIN",
                    "split": split,
                    "entity_group": entity["entity_group"],
                    "canonical_entity": entity["canonical_entity"],
                    "original_canonical_entity_key": entity_key,
                    "decision": status,
                    "selected_qid": selected_qid,
                    "confidence": decision["confidence"],
                    "reason": decision["reason"],
                    "candidate_count": len(allowed_qids),
                    "linked_entity_key": linked_entity_key,
                    "resolution_source": (
                        "GPT_REPAIR_FALLBACK_UNLINKED"
                        if is_repair_fallback
                        else "GPT"
                    ),
                    "gpt_cache_key": cache_record["cache_key"],
                }
            )

    # Original sequential path.
    if gpt_workers == 1:
        for index, (article_key, _article_value, split) in enumerate(selected_scope, 1):
            article_entities, gpt_needed = prepare_article(article_key, split)
            if not article_entities:
                maybe_email_progress(index)
                continue

            for entity_chunk in chunk_objects(gpt_needed, gpt_entities_per_request):
                payload = build_gpt_payload(
                    article_id_key=article_key,
                    split=scope_split[article_key],
                    article_context=article_context_map.get(
                        article_key, {"title": "", "context": ""}
                    ),
                    article_entities=article_entities,
                    entity_chunk=entity_chunk,
                    surface_search_records=surface_search_records,
                    wd=wd,
                    max_coentities=max_coentities,
                )
                decisions, cache_record = gpt_linker.link(payload)
                append_gpt_result_rows(
                    article_key, split, entity_chunk, decisions, cache_record
                )

            maybe_email_progress(index)

            if index % 100 == 0 or index == len(selected_scope):
                stats = gpt_linker.stats_snapshot()
                token_progress = format_token_progress(stats)
                info(
                    f"Link progress: {index}/{len(selected_scope)} articles; "
                    f"new GPT calls={stats['gpt_cache_write']}, "
                    f"cache hits={stats['gpt_cache_hit']}; "
                    f"{token_progress}"
                )

        link_rows.sort(
            key=lambda r: (r["article_id_key"], r["original_canonical_entity_key"])
        )
        return link_rows

    # Parallel path: keep only 2x workers requests in flight.
    max_inflight = max(gpt_workers, gpt_workers * 2)
    submitted_requests = 0
    completed_requests = 0
    inflight: Dict[Any, Tuple[str, str, List[Dict[str, Any]]]] = {}

    with ThreadPoolExecutor(
        max_workers=gpt_workers,
        thread_name_prefix="sid-gpt-link",
    ) as executor:

        def consume_done(done_futures: Iterable[Any]) -> None:
            nonlocal completed_requests
            for future in done_futures:
                article_key, split, entity_chunk = inflight.pop(future)
                decisions, cache_record = future.result()
                append_gpt_result_rows(
                    article_key, split, entity_chunk, decisions, cache_record
                )
                completed_requests += 1

        for index, (article_key, _article_value, split) in enumerate(selected_scope, 1):
            article_entities, gpt_needed = prepare_article(article_key, split)
            if not article_entities:
                maybe_email_progress(
                    index,
                    completed_requests=completed_requests,
                    submitted_requests=submitted_requests,
                    in_flight=len(inflight),
                )
                continue

            for entity_chunk_seq in chunk_objects(
                gpt_needed, gpt_entities_per_request
            ):
                entity_chunk = list(entity_chunk_seq)

                # Keep Wikidata payload construction on the main thread.
                payload = build_gpt_payload(
                    article_id_key=article_key,
                    split=scope_split[article_key],
                    article_context=article_context_map.get(
                        article_key, {"title": "", "context": ""}
                    ),
                    article_entities=article_entities,
                    entity_chunk=entity_chunk,
                    surface_search_records=surface_search_records,
                    wd=wd,
                    max_coentities=max_coentities,
                )

                future = executor.submit(gpt_linker.link, payload)
                inflight[future] = (article_key, split, entity_chunk)
                submitted_requests += 1

                if len(inflight) >= max_inflight:
                    done, _ = wait(
                        list(inflight.keys()),
                        return_when=FIRST_COMPLETED,
                    )
                    consume_done(done)

            maybe_email_progress(
                index,
                completed_requests=completed_requests,
                submitted_requests=submitted_requests,
                in_flight=len(inflight),
            )

            if index % 100 == 0 or index == len(selected_scope):
                stats = gpt_linker.stats_snapshot()
                token_progress = format_token_progress(stats)
                info(
                    f"Link progress: {index}/{len(selected_scope)} articles submitted; "
                    f"GPT requests completed={completed_requests}/{submitted_requests}; "
                    f"in_flight={len(inflight)}; workers={gpt_workers}; "
                    f"new GPT calls={stats['gpt_cache_write']}, "
                    f"cache hits={stats['gpt_cache_hit']}; "
                    f"{token_progress}"
                )

        while inflight:
            done, _ = wait(
                list(inflight.keys()),
                return_when=FIRST_COMPLETED,
            )
            consume_done(done)

    link_rows.sort(
        key=lambda r: (r["article_id_key"], r["original_canonical_entity_key"])
    )
    return link_rows

def build_article_linked_entities_rows(
    selected_scope: Sequence[Tuple[str, Any, str]], link_rows: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    by_article: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in link_rows:
        by_article[row["article_id_key"]].append(row)

    result: List[Dict[str, Any]] = []
    for article_key, article_value, split in selected_scope:
        rows = by_article.get(article_key, [])
        final_keys = sorted({r["linked_entity_key"] for r in rows})
        original_keys = sorted({r["original_canonical_entity_key"] for r in rows})
        counts = Counter(r["decision"] for r in rows)
        result.append(
            {
                "article_id": article_value,
                "is_train_used": split == "TRAIN",
                "split": split,
                "original_entity_count": len(original_keys),
                "linked_final_entity_count": len(final_keys),
                "linked_count": counts[DECISION_LINKED],
                "ambiguous_count": counts[DECISION_AMBIGUOUS],
                "unlinked_count": counts[DECISION_UNLINKED],
                "linked_entities": final_keys,
            }
        )
    return result


def add_link_columns_to_original_entity_rows(
    entity_df: pl.DataFrame,
    schema: ResolvedSchema,
    scope_keys: set[str],
    link_rows: Sequence[Dict[str, Any]],
) -> pl.DataFrame:
    working = entity_df.with_columns(
        pl.col(schema.entity_article_id).cast(pl.Utf8).alias("__article_id_key")
    ).filter(pl.col("__article_id_key").is_in(sorted(scope_keys)))

    if schema.canonical_key:
        working = working.with_columns(pl.col(schema.canonical_key).cast(pl.Utf8).alias("__canonical_key"))
    else:
        assert schema.entity_group and schema.canonical_entity
        working = working.with_columns(
            pl.concat_str(
                [
                    pl.col(schema.entity_group).cast(pl.Utf8),
                    pl.lit("::"),
                    pl.col(schema.canonical_entity).cast(pl.Utf8),
                ]
            ).alias("__canonical_key")
        )

    join_rows = [
        {
            "__article_id_key": r["article_id_key"],
            "__canonical_key": r["original_canonical_entity_key"],
            "link_decision": r["decision"],
            "selected_qid": r.get("selected_qid"),
            "linked_entity_key": r["linked_entity_key"],
        }
        for r in link_rows
    ]
    if join_rows:
        link_df = pl.DataFrame(join_rows)
        working = working.join(link_df, on=["__article_id_key", "__canonical_key"], how="left")
    else:
        working = working.with_columns(
            pl.lit(None).cast(pl.Utf8).alias("link_decision"),
            pl.lit(None).cast(pl.Utf8).alias("selected_qid"),
            pl.lit(None).cast(pl.Utf8).alias("linked_entity_key"),
        )
    return working.drop(["__article_id_key", "__canonical_key"])


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------


def estimate_gpt5_mini_cost(stats: Counter, model: str) -> Optional[float]:
    # Current pinned GPT-5 mini snapshot rate at script creation date (2026-08-18):
    # input $0.25/M, cached input $0.025/M, output $2.00/M.
    if not model.startswith("gpt-5-mini"):
        return None
    input_tokens = stats["new_input_tokens"]
    cached = stats["new_cached_input_tokens"]
    uncached = max(0, input_tokens - cached)
    output = stats["new_output_tokens"]
    return uncached / 1_000_000 * 0.25 + cached / 1_000_000 * 0.025 + output / 1_000_000 * 2.0


def make_summary(
    *,
    args: argparse.Namespace,
    selected_scope: Sequence[Tuple[str, Any, str]],
    pair_rows: Sequence[Dict[str, Any]],
    pairs_by_article: Dict[str, List[Dict[str, Any]]],
    surface_search_records: Dict[str, Dict[str, Any]],
    wd: WikidataClient,
    gpt_linker: Optional[GPTLinker],
    link_rows: Optional[Sequence[Dict[str, Any]]],
) -> str:
    train_articles = sum(1 for _k, _v, split in selected_scope if split == "TRAIN")
    val_articles = sum(1 for _k, _v, split in selected_scope if split == "VALIDATION")
    articles_with_entities = sum(1 for k, _v, _s in selected_scope if pairs_by_article.get(k))
    unique_keys = {
        (r["entity_group"], r["canonical_entity"], r["original_canonical_entity_key"]) for r in pair_rows
    }
    found_surfaces = sum(1 for r in surface_search_records.values() if r.get("candidates"))
    no_candidate_surfaces = len(surface_search_records) - found_surfaces

    lines = [
        "SID_Project Entity Linking Summary",
        "=================================",
        f"prompt_version                     = {PROMPT_VERSION}",
        "gpt_payload_profile                = LEAN_V2_SAFE_COMPRESSION",
        "gpt_cache_file                     = gpt_link_cache_lean_v2.jsonl",
        f"model                              = {args.model}",
        f"mode                               = {'SEARCH_ONLY' if args.search_only else 'LINK_ONLY' if args.link_only else 'SEARCH_AND_LINK'}",
        f"limit_articles                     = {args.limit_articles}",
        f"gpt_workers                        = {args.gpt_workers}",
        "",
        "[Frozen policy - NOT changed]",
        f"event_similarity_threshold         = {EVENT_ENTITY_SIMILARITY_THRESHOLD}",
        f"event_time_window_hours            = {EVENT_TIME_WINDOW_HOURS}",
        f"event_max_entity_df_ratio          = {EVENT_MAX_ENTITY_DF_RATIO}",
        "TYPE policy                        = SOFT_EVIDENCE_ONLY",
        "global surface->QID                = FORBIDDEN",
        "unresolved final key               = normalize_v2 canonical key",
        "",
        "[Scope]",
        f"article_count                      = {len(selected_scope):,}",
        f"train_article_count                = {train_articles:,}",
        f"validation_article_count           = {val_articles:,}",
        f"articles_with_entities             = {articles_with_entities:,}",
        f"articles_without_entities          = {len(selected_scope) - articles_with_entities:,}",
        f"article_entity_case_count          = {len(pair_rows):,}",
        f"unique_canonical_entity_key_count  = {len(unique_keys):,}",
        f"unique_search_surface_count        = {len(surface_search_records):,}",
        "",
        "[Wikidata]",
        f"surface_found_count                = {found_surfaces:,}",
        f"surface_no_candidate_count         = {no_candidate_surfaces:,}",
        f"search_cache_hit_this_run          = {wd.stats['search_cache_hit']:,}",
        f"search_cache_write_this_run        = {wd.stats['search_cache_write']:,}",
        f"search_http_requests_this_run      = {wd.stats['search_http_request']:,}",
        f"entity_meta_cache_write_this_run   = {wd.stats['entity_meta_cache_write']:,}",
        f"entity_meta_http_requests_this_run = {wd.stats['entity_meta_http_request']:,}",
    ]

    if gpt_linker is not None:
        estimated_cost = estimate_gpt5_mini_cost(gpt_linker.stats, args.model)
        lines += [
            "",
            "[GPT]",
            f"gpt_cache_hit_this_run             = {gpt_linker.stats['gpt_cache_hit']:,}",
            f"gpt_new_requests_this_run          = {gpt_linker.stats['gpt_cache_write']:,}",
            f"new_input_tokens                   = {gpt_linker.stats['new_input_tokens']:,}",
            f"new_cached_input_tokens            = {gpt_linker.stats['new_cached_input_tokens']:,}",
            f"new_output_tokens                  = {gpt_linker.stats['new_output_tokens']:,}",
        ]
        if estimated_cost is not None:
            lines.append(f"estimated_new_api_cost_usd         = {estimated_cost:.6f}")

    if link_rows is not None:
        counts = Counter(r["decision"] for r in link_rows)
        linked_qids = {r["selected_qid"] for r in link_rows if r.get("selected_qid")}
        no_candidate_cases = sum(1 for r in link_rows if r.get("resolution_source") == "NO_CANDIDATE")
        changed_keys = sum(
            1 for r in link_rows if r["linked_entity_key"] != r["original_canonical_entity_key"]
        )
        lines += [
            "",
            "[Link decisions]",
            f"LINKED                             = {counts[DECISION_LINKED]:,}",
            f"AMBIGUOUS                          = {counts[DECISION_AMBIGUOUS]:,}",
            f"UNLINKED                           = {counts[DECISION_UNLINKED]:,}",
            f"no_candidate_cases                 = {no_candidate_cases:,}",
            f"distinct_linked_qid_count          = {len(linked_qids):,}",
            f"final_key_changed_cases            = {changed_keys:,}",
        ]

    lines += [
        "",
        "[Important]",
        "This script creates Entity Linking outputs only.",
        "It does NOT change/rebuild the frozen Event algorithm.",
        "After review, feed article_linked_entities.parquet into the existing Event builder connector and",
        "recompute Train DF -> High-DF -> IDF -> 72h candidates -> weighted Jaccard >= 0.3 -> Events.",
    ]
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full Train+Validation Wikidata + GPT Entity Linking")

    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit-articles", type=int, default=None)
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--link-only", action="store_true")
    parser.add_argument(
        "--notify-email",
        default=None,
        help="Optional recipient for progress/completion/error email notifications.",
    )
    parser.add_argument(
        "--notify-every-articles",
        type=int,
        default=1000,
        help="Send progress email every N article positions (default: 1000).",
    )
    parser.add_argument("--smtp-host", default="smtp.gmail.com")
    parser.add_argument("--smtp-port", type=int, default=587)

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["minimal", "low", "medium", "high"],
    )
    parser.add_argument("--gpt-entities-per-request", type=int, default=8)
    parser.add_argument(
        "--gpt-workers",
        type=int,
        default=1,
        help="Parallel OpenAI request workers. 1=sequential; recommended start=3.",
    )
    parser.add_argument("--openai-max-output-tokens", type=int, default=3000)
    parser.add_argument("--max-coentities", type=int, default=30)
    parser.add_argument("--article-context-chars", type=int, default=1800)

    parser.add_argument("--wikidata-api", default=DEFAULT_WIKIDATA_API)
    parser.add_argument("--wikidata-limit-per-lang", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--wikidata-delay", type=float, default=0.05)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--wikidata-user-agent",
        default=os.getenv("WIKIDATA_USER_AGENT", "SID_Project-EntityLinking/1.0 (research; local script)"),
    )

    # Optional schema overrides. Usually not needed; the script auto-detects common names.
    parser.add_argument("--entity-article-id-column", default=None)
    parser.add_argument("--article-article-id-column", default=None)
    parser.add_argument("--entity-group-column", default=None)
    parser.add_argument("--canonical-entity-column", default=None)
    parser.add_argument("--canonical-key-column", default=None)
    parser.add_argument("--title-column", default=None)
    parser.add_argument("--context-columns", nargs="*", default=None)

    args = parser.parse_args()

    if args.search_only and args.link_only:
        parser.error("--search-only and --link-only are mutually exclusive")
    if not 1 <= args.gpt_entities_per_request <= 8:
        parser.error("--gpt-entities-per-request must be between 1 and 8 (recommended: 5-8)")
    if not 1 <= args.gpt_workers <= 8:
        parser.error("--gpt-workers must be between 1 and 8 (recommended: start with 3)")
    if args.wikidata_limit_per_lang < 1:
        parser.error("--wikidata-limit-per-lang must be >= 1")
    if args.max_candidates < 1:
        parser.error("--max-candidates must be >= 1")
    if args.max_retries < 1:
        parser.error("--max-retries must be >= 1")
    if args.notify_every_articles < 1:
        parser.error("--notify-every-articles must be >= 1")
    if args.smtp_port < 1:
        parser.error("--smtp-port must be >= 1")

    return args


def main() -> None:
    global _EMAIL_NOTIFIER_FOR_FATAL_ERROR

    args = parse_args()
    email_notifier = build_email_notifier(args)
    _EMAIL_NOTIFIER_FOR_FATAL_ERROR = email_notifier

    if email_notifier is not None:
        email_notifier.send(
            "[SID Entity Linking] STARTED",
            "SID_Project Entity Linking started.\n"
            f"Progress email interval: every {args.notify_every_articles:,} articles.\n"
            f"Workers: {args.gpt_workers}\n"
            f"Model: {args.model}\n"
            "Existing Lean v2 GPT cache will be reused."
        )

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    # Frozen snapshot protection: output must never be inside baseline/normalize_v2.
    experiments_dir = input_dir.parent.parent
    frozen_dirs = [experiments_dir / "baseline", experiments_dir / "normalize_v2"]
    for frozen in frozen_dirs:
        if path_is_within(output_dir, frozen):
            raise RuntimeError(
                f"Refusing to write Entity Linking outputs inside frozen snapshot: {frozen}. "
                f"Choose a separate --output-dir (default: {DEFAULT_OUTPUT_DIR})."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    article_entities_path = input_dir / "article_entities.parquet"
    articles_base_path = input_dir / "articles_base.parquet"
    train_scope_path = input_dir / "article_master.parquet"
    validation_article_master_path = input_dir / "validation_article_master.parquet"
    validation_scope_path = input_dir / "validation_article_events.parquet"

    required = [article_entities_path, articles_base_path, train_scope_path, validation_scope_path]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required normalize_v2 read-only inputs are missing:\n" + "\n".join(f"  - {p}" for p in missing)
        )

    info(f"Reading frozen normalize_v2 inputs from: {input_dir}")
    entity_df = pl.read_parquet(article_entities_path)
    article_df = pl.read_parquet(articles_base_path)
    # Some snapshots keep Validation article text/context in a separate master file.
    # Merge it read-only when present so Train and Validation both receive article context.
    if validation_article_master_path.exists():
        validation_article_df = pl.read_parquet(validation_article_master_path)
        article_df = pl.concat([article_df, validation_article_df], how="diagonal_relaxed")
        info(f"Merged Validation article context from: {validation_article_master_path}")
    else:
        warn(
            f"Optional Validation article master not found: {validation_article_master_path}. "
            "Continuing with articles_base.parquet; missing Validation context will be reported."
        )
    schema = resolve_schema(entity_df, article_df, args)

    info(
        "Resolved schema: "
        f"entity_article_id={schema.entity_article_id}, entity_group={schema.entity_group}, "
        f"canonical_entity={schema.canonical_entity}, canonical_key={schema.canonical_key}, "
        f"title={schema.title}, context_columns={schema.context_columns}"
    )

    train_rows = read_scope_ids(train_scope_path)
    validation_rows = read_scope_ids(validation_scope_path)
    selected_scope, _is_train_map = choose_scope(train_rows, validation_rows, args.limit_articles)
    scope_keys = {k for k, _v, _split in selected_scope}

    train_count = sum(1 for _k, _v, split in selected_scope if split == "TRAIN")
    val_count = sum(1 for _k, _v, split in selected_scope if split == "VALIDATION")
    info(f"Selected article scope: total={len(selected_scope)}, train={train_count}, validation={val_count}")

    pair_rows, pairs_by_article = build_normalized_entity_pairs(entity_df, schema, scope_keys)
    article_context_map = build_article_context_map(
        article_df, schema, scope_keys, context_chars=args.article_context_chars
    )
    missing_context_articles = sum(1 for k, _v, _s in selected_scope if k not in article_context_map)
    if missing_context_articles:
        warn(f"{missing_context_articles} scoped articles were not found in articles_base; GPT gets empty title/context for them.")

    unique_surfaces = collect_unique_surfaces(pair_rows)
    unique_entity_keys = sorted(
        {
            (r["entity_group"], r["canonical_entity"], r["original_canonical_entity_key"])
            for r in pair_rows
        },
        key=lambda x: x[2],
    )
    info(
        f"Entity scope: article/entity cases={len(pair_rows)}, "
        f"unique canonical keys={len(unique_entity_keys)}, unique search surfaces={len(unique_surfaces)}"
    )

    wikidata_search_cache = JsonlCache(output_dir / "wikidata_search_cache.jsonl")
    wikidata_entity_cache = JsonlCache(output_dir / "wikidata_entity_cache.jsonl")
    gpt_cache = JsonlCache(output_dir / "gpt_link_cache_lean_v2.jsonl")
    info(
        "Lean v2 GPT cache: gpt_link_cache_lean_v2.jsonl "
        "(old gpt_link_cache.jsonl is preserved; Wikidata caches are reused)"
    )

    wd = WikidataClient(
        api_url=args.wikidata_api,
        user_agent=args.wikidata_user_agent,
        search_cache=wikidata_search_cache,
        entity_cache=wikidata_entity_cache,
        langs=DEFAULT_WIKIDATA_LANGS,
        limit_per_lang=args.wikidata_limit_per_lang,
        max_candidates=args.max_candidates,
        timeout_s=args.request_timeout,
        delay_s=args.wikidata_delay,
        max_retries=args.max_retries,
    )

    # 1) Surface-level candidate search. Same surface cache is reused across every article.
    surface_search_records: Dict[str, Dict[str, Any]] = {}
    allow_wikidata_network = not args.link_only
    for i, surface in enumerate(unique_surfaces, 1):
        record = wd.search_surface(surface, allow_network=allow_wikidata_network)
        surface_search_records[normalize_surface_cache_key(surface)] = record
        if i % 250 == 0 or i == len(unique_surfaces):
            info(
                f"Wikidata search progress: {i}/{len(unique_surfaces)} surfaces; "
                f"new={wd.stats['search_cache_write']}, cache_hits={wd.stats['search_cache_hit']}"
            )

    # 2) Batch metadata enrichment for all candidate QIDs and their P31/P279 context.
    candidate_qids = collect_all_candidate_qids(surface_search_records)
    info(f"Unique Wikidata candidate QIDs: {len(candidate_qids)}")
    wd.enrich_candidate_type_metadata(candidate_qids, allow_network=allow_wikidata_network)

    # 3) Candidate audit parquet (safe to overwrite inside the NEW experiment directory).
    candidate_rows = build_wikidata_candidates_rows(unique_entity_keys, surface_search_records, wd)
    candidate_df = pl.DataFrame(candidate_rows) if candidate_rows else pl.DataFrame(
        schema={
            "entity_group": pl.Utf8,
            "canonical_entity": pl.Utf8,
            "original_canonical_entity_key": pl.Utf8,
            "candidate_index": pl.Int64,
            "qid": pl.Utf8,
            "label": pl.Utf8,
            "description": pl.Utf8,
            "type_diagnostic": pl.Utf8,
            "search_hits_json": pl.Utf8,
            "instance_of_json": pl.Utf8,
            "direct_subclass_of_json": pl.Utf8,
        }
    )
    atomic_write_parquet(candidate_df, output_dir / "wikidata_candidates.parquet")

    if args.search_only:
        summary = make_summary(
            args=args,
            selected_scope=selected_scope,
            pair_rows=pair_rows,
            pairs_by_article=pairs_by_article,
            surface_search_records=surface_search_records,
            wd=wd,
            gpt_linker=None,
            link_rows=None,
        )
        atomic_write_text(output_dir / "entity_linking_summary.txt", summary)
        print(summary)
        info("Search-only completed. No OpenAI API call was made.")
        return

    # 4) GPT article-context linking.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in PowerShell before linking. "
            "Search-only can run without the OpenAI key."
        )

    openai_client = OpenAI()
    gpt_linker = GPTLinker(
        client=openai_client,
        cache=gpt_cache,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.openai_max_output_tokens,
        max_retries=args.max_retries,
    )

    link_rows = build_article_entity_links(
        selected_scope=selected_scope,
        pairs_by_article=pairs_by_article,
        article_context_map=article_context_map,
        surface_search_records=surface_search_records,
        wd=wd,
        gpt_linker=gpt_linker,
        gpt_entities_per_request=args.gpt_entities_per_request,
        max_coentities=args.max_coentities,
        gpt_workers=args.gpt_workers,
        email_notifier=email_notifier,
        notify_every_articles=args.notify_every_articles,
    )

    # 5) Final link audit table.
    links_df = pl.DataFrame(link_rows) if link_rows else pl.DataFrame(
        schema={
            "article_id": pl.Utf8,
            "article_id_key": pl.Utf8,
            "is_train_used": pl.Boolean,
            "split": pl.Utf8,
            "entity_group": pl.Utf8,
            "canonical_entity": pl.Utf8,
            "original_canonical_entity_key": pl.Utf8,
            "decision": pl.Utf8,
            "selected_qid": pl.Utf8,
            "confidence": pl.Utf8,
            "reason": pl.Utf8,
            "candidate_count": pl.Int64,
            "linked_entity_key": pl.Utf8,
            "resolution_source": pl.Utf8,
        }
    )
    # article_id_key is internal resume/join plumbing; keep it in audit output for traceability.
    atomic_write_parquet(links_df, output_dir / "article_entity_links.parquet")

    # 6) Original normalize_v2 entity rows + link columns. Source file remains untouched.
    enriched_entity_df = add_link_columns_to_original_entity_rows(
        entity_df, schema, scope_keys, link_rows
    )
    atomic_write_parquet(enriched_entity_df, output_dir / "article_entities_linked.parquet")

    # 7) Event connector-ready article -> set[final entity keys], including empty-entity articles.
    article_linked_rows = build_article_linked_entities_rows(selected_scope, link_rows)
    article_linked_df = pl.DataFrame(article_linked_rows)
    atomic_write_parquet(article_linked_df, output_dir / "article_linked_entities.parquet")

    # 8) Summary.
    summary = make_summary(
        args=args,
        selected_scope=selected_scope,
        pair_rows=pair_rows,
        pairs_by_article=pairs_by_article,
        surface_search_records=surface_search_records,
        wd=wd,
        gpt_linker=gpt_linker,
        link_rows=link_rows,
    )
    atomic_write_text(output_dir / "entity_linking_summary.txt", summary)
    print(summary)

    info(f"Completed. Outputs written only under: {output_dir}")
    info("Frozen baseline/normalize_v2 inputs were read-only and were not modified.")

    if email_notifier is not None:
        final_stats = gpt_linker.stats_snapshot()
        final_cost = estimate_gpt5_mini_cost(final_stats, args.model)
        completion_lines = [
            "SID_Project Entity Linking COMPLETED",
            "",
            f"articles = {len(selected_scope):,}/{len(selected_scope):,}",
            f"output_dir = {output_dir}",
            "",
            "Final files:",
            "- article_entity_links.parquet",
            "- article_entities_linked.parquet",
            "- article_linked_entities.parquet",
            "- entity_linking_summary.txt",
            "- wikidata_candidates.parquet",
            "",
            f"new GPT cache writes this run = {final_stats['gpt_cache_write']:,}",
            f"GPT cache hits this run = {final_stats['gpt_cache_hit']:,}",
        ]
        if final_cost is not None:
            completion_lines.append(
                f"estimated API cost this run = ${final_cost:.4f}"
            )
        completion_lines += [
            "",
            "Entity Linking is complete.",
            "Event rebuilding is NOT included in this runner.",
        ]
        email_notifier.send(
            "[SID Entity Linking] COMPLETED",
            "\n".join(completion_lines),
        )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        notifier = _EMAIL_NOTIFIER_FOR_FATAL_ERROR
        if notifier is not None and not isinstance(exc, KeyboardInterrupt):
            error_text = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            notifier.send(
                "[SID Entity Linking] FAILED",
                "SID_Project Entity Linking stopped with an error.\n\n"
                + error_text[-7000:],
            )
        raise
