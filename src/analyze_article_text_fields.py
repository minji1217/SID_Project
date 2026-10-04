from __future__ import annotations

"""
EB-NeRD article 텍스트 컬럼 가용성 분석
=======================================

이 파일의 목적
--------------
`articles.parquet`의 `title`, `subtitle`, `body` 중 어떤 컬럼을 모델 입력으로
쓸 수 있는지 판단하려면 "컬럼이 존재하는가"가 아니라 "실제 텍스트가 들어 있는가"를
봐야 한다. null이 아니어도 빈 문자열이거나 공백만 있는 경우가 있고, 값이 있어도
10자짜리 캡션이면 본문으로 쓸 수 없다.

그래서 이 스크립트는 세 가지를 구분해서 센다.

    null        값이 None
    blank       null은 아니지만 strip() 결과가 빈 문자열
    present     null도 blank도 아님 (= 실제 텍스트가 있음)

그리고 present인 값에 대해서만 길이 분포를 계산한다. blank를 길이 0으로 섞으면
분포가 아래로 끌려 내려가 "본문이 짧다"와 "본문이 없다"를 구분할 수 없게 된다.

출력
----
    text_field_analysis.json          전체 결과
    sample_body_empty_subtitle_present.csv
    sample_subtitle_and_body_present.csv
    sample_subtitle_short_under20.csv
    sample_subtitle_long_50plus.csv

입력 parquet은 읽기 전용으로만 연다. 절대 수정하지 않는다.

사용법
------
    python -m src.analyze_article_text_fields \
        --articles data/articles.parquet \
        --output-dir outputs/text_field_analysis
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


TEXT_COLUMNS = ("title", "subtitle", "body")
REQUIRED_COLUMNS = ("article_id",) + TEXT_COLUMNS

# 길이 임계값
SHORT_THRESHOLDS = (10, 20, 50)
CHAR_PERCENTILES = (25, 50, 75, 90, 95, 99)
WORD_PERCENTILES = (50, 90, 95)


# ============================================================
# 유틸
# ============================================================

def pct(numerator: int, denominator: int) -> float:
    """0 나눗셈을 피하면서 백분율을 계산한다."""
    if denominator == 0:
        return float("nan")
    return 100.0 * numerator / denominator


def fmt_count(value: int, total: int) -> str:
    return f"{value:>9,}  ({pct(value, total):5.2f}%)"


def render_table(headers: list[str], rows: list[list[str]], indent: str = "  ") -> str:
    """외부 의존성 없이 고정폭 표를 그린다."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(char: str) -> str:
        return indent + "-+-".join(char * w for w in widths)

    def fmt(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            # 첫 컬럼은 좌측 정렬(레이블), 나머지는 우측 정렬(숫자)
            out.append(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]))
        return indent + " | ".join(out)

    parts = [fmt(headers), line("-")]
    parts.extend(fmt(r) for r in rows)
    return "\n".join(parts)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# 입력
# ============================================================

def load_articles(path: Path) -> pl.DataFrame:
    """parquet을 읽기 전용으로 연다. 필요한 컬럼만 가져온다."""
    if not path.exists():
        raise FileNotFoundError(f"articles parquet을 찾지 못했습니다: {path}")

    available = pl.read_parquet_schema(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in available]
    if missing:
        raise KeyError(
            f"{path}에 필요한 컬럼이 없습니다: {missing}. "
            f"존재하는 컬럼: {sorted(available)}"
        )

    frame = pl.read_parquet(path, columns=list(REQUIRED_COLUMNS))
    return frame.with_columns(
        [pl.col(c).cast(pl.Utf8, strict=False) for c in TEXT_COLUMNS]
    )


def build_masks(frame: pl.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """컬럼마다 null / blank / present 마스크와 길이 배열을 만든다.

    stripped는 앞뒤 공백을 제거한 문자열이며, 길이·단어 수·비교는 모두 이 값을 쓴다.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for col in TEXT_COLUMNS:
        raw = frame[col].to_list()
        is_null = np.array([v is None for v in raw], dtype=bool)
        stripped = np.array(
            [("" if v is None else str(v).strip()) for v in raw], dtype=object
        )
        is_blank = (~is_null) & np.array([s == "" for s in stripped], dtype=bool)
        is_present = ~is_null & ~is_blank
        char_len = np.array([len(s) for s in stripped], dtype=np.int64)
        word_len = np.array([len(s.split()) for s in stripped], dtype=np.int64)
        out[col] = {
            "stripped": stripped,
            "null": is_null,
            "blank": is_blank,
            "present": is_present,
            "char_len": char_len,
            "word_len": word_len,
        }
    return out


# ============================================================
# 1. 컬럼별 가용성 + 길이 분포
# ============================================================

def analyse_column(col: str, m: dict[str, np.ndarray], total: int) -> dict[str, Any]:
    present = m["present"]
    n_present = int(present.sum())
    chars = m["char_len"][present]
    words = m["word_len"][present]

    stats: dict[str, Any] = {
        "total_articles": total,
        "null": {"count": int(m["null"].sum()), "pct": pct(int(m["null"].sum()), total)},
        "blank_or_whitespace": {
            "count": int(m["blank"].sum()),
            "pct": pct(int(m["blank"].sum()), total),
        },
        "present": {"count": n_present, "pct": pct(n_present, total)},
        "_note": "아래 길이 통계와 짧은 텍스트 비율은 present인 기사만 대상으로 한다.",
    }

    if n_present == 0:
        stats["char_stats"] = None
        stats["word_stats"] = None
        stats["short_text"] = None
        return stats

    char_stats = {
        "mean": float(chars.mean()),
        "std": float(chars.std(ddof=1)) if n_present > 1 else float("nan"),
        "min": int(chars.min()),
        "max": int(chars.max()),
    }
    for p in CHAR_PERCENTILES:
        char_stats[f"p{p}"] = float(np.percentile(chars, p))
    stats["char_stats"] = char_stats

    word_stats = {"mean": float(words.mean()), "max": int(words.max())}
    for p in WORD_PERCENTILES:
        word_stats[f"p{p}"] = float(np.percentile(words, p))
    stats["word_stats"] = word_stats

    short = {}
    for t in SHORT_THRESHOLDS:
        c = int((chars < t).sum())
        short[f"under_{t}_chars"] = {
            "count": c,
            "pct_of_present": pct(c, n_present),
            "pct_of_all": pct(c, total),
        }
    stats["short_text"] = short
    return stats


def print_column_tables(per_column: dict[str, dict[str, Any]], total: int) -> None:
    section("1. 컬럼별 텍스트 가용성")
    print(f"\n  전체 기사 수: {total:,}\n")
    rows = []
    for col in TEXT_COLUMNS:
        s = per_column[col]
        rows.append([
            col,
            f"{s['null']['count']:,}",
            f"{s['null']['pct']:.2f}%",
            f"{s['blank_or_whitespace']['count']:,}",
            f"{s['blank_or_whitespace']['pct']:.2f}%",
            f"{s['present']['count']:,}",
            f"{s['present']['pct']:.2f}%",
        ])
    print(render_table(
        ["컬럼", "null", "null %", "blank", "blank %", "present", "present %"], rows))

    section("2. 문자 수 분포 (present인 기사만)")
    rows = []
    for col in TEXT_COLUMNS:
        cs = per_column[col]["char_stats"]
        if cs is None:
            rows.append([col] + ["–"] * 10)
            continue
        rows.append([
            col, f"{cs['mean']:.1f}", f"{cs['std']:.1f}", f"{cs['min']:,}",
            f"{cs['p25']:.0f}", f"{cs['p50']:.0f}", f"{cs['p75']:.0f}",
            f"{cs['p90']:.0f}", f"{cs['p95']:.0f}", f"{cs['p99']:.0f}", f"{cs['max']:,}",
        ])
    print(render_table(
        ["컬럼", "mean", "std", "min", "p25", "p50", "p75", "p90", "p95", "p99", "max"],
        rows))

    section("3. 단어 수 분포 (present인 기사만)")
    rows = []
    for col in TEXT_COLUMNS:
        ws = per_column[col]["word_stats"]
        if ws is None:
            rows.append([col] + ["–"] * 5)
            continue
        rows.append([col, f"{ws['mean']:.1f}", f"{ws['p50']:.0f}",
                     f"{ws['p90']:.0f}", f"{ws['p95']:.0f}", f"{ws['max']:,}"])
    print(render_table(["컬럼", "mean", "p50", "p90", "p95", "max"], rows))

    section("4. 짧은 텍스트 비율 (분모 = 해당 컬럼이 present인 기사)")
    rows = []
    for col in TEXT_COLUMNS:
        sh = per_column[col]["short_text"]
        if sh is None:
            rows.append([col] + ["–"] * 6)
            continue
        cells = [col]
        for t in SHORT_THRESHOLDS:
            d = sh[f"under_{t}_chars"]
            cells.extend([f"{d['count']:,}", f"{d['pct_of_present']:.2f}%"])
        rows.append(cells)
    print(render_table(
        ["컬럼", "<10자", "<10자 %", "<20자", "<20자 %", "<50자", "<50자 %"], rows))


# ============================================================
# 2. subtitle / body 전용 분석
# ============================================================

def analyse_subtitle(masks: dict[str, dict[str, np.ndarray]], total: int) -> dict[str, Any]:
    sub, tit = masks["subtitle"], masks["title"]
    present = sub["present"]
    n_present = int(present.sum())
    chars = sub["char_len"]

    both = present & tit["present"]
    s_txt, t_txt = sub["stripped"], tit["stripped"]

    exact = np.zeros(total, dtype=bool)
    contains = np.zeros(total, dtype=bool)
    for i in np.flatnonzero(both):
        s, t = s_txt[i], t_txt[i]
        if s == t:
            exact[i] = True
            contains[i] = True
        elif s in t or t in s:
            contains[i] = True

    n_exact = int(exact.sum())
    n_contains = int(contains.sum())
    n_contains_only = n_contains - n_exact
    ge20 = int((present & (chars >= 20)).sum())
    ge50 = int((present & (chars >= 50)).sum())

    return {
        "present": {"count": n_present, "pct_of_all": pct(n_present, total)},
        "identical_to_title": {
            "count": n_exact,
            "pct_of_present": pct(n_exact, n_present),
            "definition": "strip() 후 title과 완전히 동일",
        },
        "contained_either_direction": {
            "count": n_contains,
            "pct_of_present": pct(n_contains, n_present),
            "definition": "subtitle이 title에 포함되거나 title이 subtitle에 포함 (동일한 경우 포함)",
        },
        "contained_excluding_identical": {
            "count": n_contains_only,
            "pct_of_present": pct(n_contains_only, n_present),
            "definition": "위 조건 중 완전히 동일한 경우를 제외",
        },
        "length_ge_20": {"count": ge20, "pct_of_present": pct(ge20, n_present),
                         "pct_of_all": pct(ge20, total)},
        "length_ge_50": {"count": ge50, "pct_of_present": pct(ge50, n_present),
                         "pct_of_all": pct(ge50, total)},
        "_denominator_note": "pct_of_present의 분모는 subtitle이 present인 기사 수다.",
    }


def analyse_body(masks: dict[str, dict[str, np.ndarray]], total: int) -> dict[str, Any]:
    body = masks["body"]
    present = body["present"]
    n_present = int(present.sum())
    chars = body["char_len"]

    lt20 = int((present & (chars < 20)).sum())
    ge50 = int((present & (chars >= 50)).sum())
    ge100 = int((present & (chars >= 100)).sum())
    return {
        "present": {"count": n_present, "pct_of_all": pct(n_present, total)},
        "present_but_under_20": {"count": lt20, "pct_of_present": pct(lt20, n_present),
                                 "pct_of_all": pct(lt20, total)},
        "length_ge_50": {"count": ge50, "pct_of_present": pct(ge50, n_present),
                         "pct_of_all": pct(ge50, total)},
        "length_ge_100": {"count": ge100, "pct_of_present": pct(ge100, n_present),
                          "pct_of_all": pct(ge100, total)},
        "_denominator_note": "pct_of_present의 분모는 body가 present인 기사 수다.",
    }


def print_field_detail(sub: dict[str, Any], body: dict[str, Any]) -> None:
    section("5. subtitle 상세")
    n = sub["present"]["count"]
    print(f"\n  subtitle present: {sub['present']['count']:,} "
          f"({sub['present']['pct_of_all']:.2f}% of all)")
    print(f"  아래 % 분모 = subtitle present {n:,}건\n")
    rows = [
        ["title과 완전히 동일", f"{sub['identical_to_title']['count']:,}",
         f"{sub['identical_to_title']['pct_of_present']:.2f}%"],
        ["포함 관계 (동일 포함)", f"{sub['contained_either_direction']['count']:,}",
         f"{sub['contained_either_direction']['pct_of_present']:.2f}%"],
        ["포함 관계 (동일 제외)", f"{sub['contained_excluding_identical']['count']:,}",
         f"{sub['contained_excluding_identical']['pct_of_present']:.2f}%"],
        ["20자 이상", f"{sub['length_ge_20']['count']:,}",
         f"{sub['length_ge_20']['pct_of_present']:.2f}%"],
        ["50자 이상", f"{sub['length_ge_50']['count']:,}",
         f"{sub['length_ge_50']['pct_of_present']:.2f}%"],
    ]
    print(render_table(["항목", "기사 수", "비율"], rows))

    section("6. body 상세")
    nb = body["present"]["count"]
    print(f"\n  body present: {body['present']['count']:,} "
          f"({body['present']['pct_of_all']:.2f}% of all)")
    print(f"  아래 % 분모 = body present {nb:,}건\n")
    rows = [
        ["present이지만 20자 미만", f"{body['present_but_under_20']['count']:,}",
         f"{body['present_but_under_20']['pct_of_present']:.2f}%"],
        ["50자 이상", f"{body['length_ge_50']['count']:,}",
         f"{body['length_ge_50']['pct_of_present']:.2f}%"],
        ["100자 이상", f"{body['length_ge_100']['count']:,}",
         f"{body['length_ge_100']['pct_of_present']:.2f}%"],
    ]
    print(render_table(["항목", "기사 수", "비율"], rows))


# ============================================================
# 3. 조합 coverage
# ============================================================

def analyse_coverage(masks: dict[str, dict[str, np.ndarray]], total: int) -> dict[str, Any]:
    t, s, b = masks["title"]["present"], masks["subtitle"]["present"], masks["body"]["present"]

    named = {
        "title_only": (t & ~s & ~b, "title만 present (subtitle·body 모두 없음)"),
        "title_and_subtitle": (t & s, "title과 subtitle이 모두 present (body는 무관)"),
        "title_and_body": (t & b, "title과 body가 모두 present (subtitle은 무관)"),
        "title_subtitle_body": (t & s & b, "셋 다 present"),
        "subtitle_without_body": (s & ~b, "subtitle은 present이고 body는 없음"),
        "body_without_subtitle": (b & ~s, "body는 present이고 subtitle은 없음"),
    }
    result = {
        k: {"count": int(m.sum()), "pct_of_all": pct(int(m.sum()), total), "definition": d}
        for k, (m, d) in named.items()
    }

    # 겹치지 않는 8개 조합. 위 named는 서로 겹치므로 합이 전체와 맞지 않는데,
    # 이 행렬은 상호배타라 합이 정확히 전체 기사 수가 된다.
    exclusive = {}
    for ti in (True, False):
        for si in (True, False):
            for bi in (True, False):
                mask = (t if ti else ~t) & (s if si else ~s) & (b if bi else ~b)
                key = "+".join(
                    n for n, f in (("title", ti), ("subtitle", si), ("body", bi)) if f
                ) or "none"
                exclusive[key] = {"count": int(mask.sum()),
                                  "pct_of_all": pct(int(mask.sum()), total)}
    result["_exclusive_matrix"] = exclusive
    result["_note"] = (
        "named 항목은 서로 겹친다 (예: title_subtitle_body는 title_and_subtitle에도 포함). "
        "_exclusive_matrix는 상호배타이며 합이 전체 기사 수와 같다."
    )
    return result


def print_coverage(cov: dict[str, Any], total: int) -> None:
    section("7. 조합 coverage")
    print("\n  [A] 요청한 조합 — 서로 겹칠 수 있음\n")
    rows = []
    for key in ("title_only", "title_and_subtitle", "title_and_body",
                "title_subtitle_body", "subtitle_without_body", "body_without_subtitle"):
        d = cov[key]
        rows.append([key, f"{d['count']:,}", f"{d['pct_of_all']:.2f}%", d["definition"]])
    print(render_table(["조합", "기사 수", "비율", "정의"], rows))

    print("\n  [B] 상호배타 행렬 — 합계가 전체와 일치\n")
    rows = []
    sub_total = 0
    for key, d in cov["_exclusive_matrix"].items():
        sub_total += d["count"]
        rows.append([key, f"{d['count']:,}", f"{d['pct_of_all']:.2f}%"])
    rows.append(["합계", f"{sub_total:,}", f"{pct(sub_total, total):.2f}%"])
    print(render_table(["present 조합", "기사 수", "비율"], rows))


# ============================================================
# 4. 샘플 추출
# ============================================================

def write_samples(frame: pl.DataFrame, masks: dict[str, dict[str, np.ndarray]],
                  out_dir: Path, seed: int, size: int, preview: int) -> dict[str, Any]:
    t, s, b = masks["title"]["present"], masks["subtitle"]["present"], masks["body"]["present"]
    s_chars = masks["subtitle"]["char_len"]

    specs = {
        "sample_body_empty_subtitle_present": (
            ~b & s, "body가 비어있고(null 또는 공백) subtitle은 present"),
        "sample_subtitle_and_body_present": (
            s & b, "subtitle과 body가 모두 present"),
        "sample_subtitle_short_under20": (
            s & (s_chars < 20), "subtitle이 present이지만 20자 미만"),
        "sample_subtitle_long_50plus": (
            s & (s_chars >= 50), "subtitle이 50자 이상"),
    }

    rng = np.random.default_rng(seed)
    article_id = frame["article_id"].to_list()
    info: dict[str, Any] = {}

    for name, (mask, desc) in specs.items():
        idx = np.flatnonzero(mask)
        n_match = int(idx.size)
        take = idx if n_match <= size else np.sort(rng.choice(idx, size=size, replace=False))

        rows = []
        for i in take:
            body_txt = masks["body"]["stripped"][i]
            truncated = len(body_txt) > preview
            rows.append({
                "article_id": str(article_id[i]),
                "title": masks["title"]["stripped"][i],
                "subtitle": masks["subtitle"]["stripped"][i],
                "body": body_txt[:preview] + ("…" if truncated else ""),
                "body_char_len_full": int(masks["body"]["char_len"][i]),
                "body_truncated": truncated,
            })

        path = out_dir / f"{name}.csv"
        schema = {"article_id": pl.Utf8, "title": pl.Utf8, "subtitle": pl.Utf8,
                  "body": pl.Utf8, "body_char_len_full": pl.Int64,
                  "body_truncated": pl.Boolean}
        pl.DataFrame(rows, schema=schema).write_csv(path)
        info[name] = {"matching_articles": n_match, "sampled": int(take.size),
                      "definition": desc, "file": path.name}
        print(f"  {name:42s} 대상 {n_match:>7,}건 → {int(take.size):>2}건 저장")

    return info


# ============================================================
# main
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EB-NeRD articles.parquet의 title/subtitle/body 텍스트 가용성 분석 "
                    "(입력 parquet은 읽기 전용)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--articles", type=Path, required=True,
                        help="articles.parquet 경로")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="JSON과 샘플 CSV를 저장할 폴더")
    parser.add_argument("--seed", type=int, default=42, help="샘플 추출 seed")
    parser.add_argument("--sample-size", type=int, default=30,
                        help="샘플 CSV 하나당 최대 기사 수")
    parser.add_argument("--body-preview", type=int, default=500,
                        help="샘플 CSV의 body를 앞에서 몇 자까지 보일지")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_size <= 0 or args.body_preview <= 0:
        raise SystemExit("--sample-size와 --body-preview는 1 이상이어야 합니다.")

    # 입력 문제는 사용자 실수이므로 traceback 대신 한 줄로 알려준다.
    try:
        frame = load_articles(args.articles)
    except (FileNotFoundError, KeyError) as exc:
        raise SystemExit(f"입력 오류: {exc.args[0] if exc.args else exc}")

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"입력  : {args.articles}  (read-only)")
    print(f"출력  : {out_dir}")
    print(f"seed  : {args.seed}")

    total = frame.height
    print(f"기사 수: {total:,}")

    masks = build_masks(frame)
    per_column = {c: analyse_column(c, masks[c], total) for c in TEXT_COLUMNS}
    subtitle_detail = analyse_subtitle(masks, total)
    body_detail = analyse_body(masks, total)
    coverage = analyse_coverage(masks, total)

    print_column_tables(per_column, total)
    print_field_detail(subtitle_detail, body_detail)
    print_coverage(coverage, total)

    section("8. 샘플 CSV")
    print()
    samples = write_samples(frame, masks, out_dir, args.seed,
                            args.sample_size, args.body_preview)

    report = {
        "input": {
            "articles_parquet": str(args.articles.resolve()),
            "read_only": True,
            "total_articles": total,
        },
        "definitions": {
            "null": "값이 None",
            "blank_or_whitespace": "null은 아니지만 strip() 결과가 빈 문자열",
            "present": "null도 blank도 아님 (실제 텍스트 있음)",
            "char_len": "strip()한 문자열의 문자 수",
            "word_len": "strip()한 문자열을 공백으로 나눈 토큰 수",
            "length_stats_scope": "길이 통계와 짧은 텍스트 비율은 present인 기사만 대상",
        },
        "per_column": per_column,
        "subtitle_detail": subtitle_detail,
        "body_detail": body_detail,
        "coverage": coverage,
        "samples": {"seed": args.seed, "max_per_file": args.sample_size,
                    "body_preview_chars": args.body_preview, "files": samples},
    }

    json_path = out_dir / "text_field_analysis.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print()
    print(f"JSON 저장: {json_path}")
    print(f"완료. 입력 parquet은 수정하지 않았습니다.")


if __name__ == "__main__":
    sys.exit(main())
