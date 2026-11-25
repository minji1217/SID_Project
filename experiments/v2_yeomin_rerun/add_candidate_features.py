"""1pos4neg 파일(train, validation_half, test)에 후보 특징 컬럼 2개를 붙여 새 폴더에 저장한다. 원래 파일은 그대로 둔다.

    python experiments/v2_yeomin_rerun/add_candidate_features.py --run h      # v1 / h / r

붙이는 컬럼 (후보 5개와 같은 순서의 float32 5개)
  candidate_logpop = log(1 + 노출 직전 N시간(기본 24h) 동안 전체 사용자 클릭 수)
                     train + validation behaviors 의 클릭 중 [노출 시각 − N시간, 노출 시각) 만 센다 (미래 정보 없음)
  candidate_logage = log(1 + max(0, 노출 시각 − 발행 시각) 시간)   발행 시각은 articles.parquet

popularity_fusion.py 의 logpop · logage 와 같은 규칙이다.
주의: 클릭 기록이 train behaviors 시작(2023-05-18 07:00)부터라, train 첫 24시간 행은 클릭 수가 실제보다 적게 세진다 (validation·test 는 해당 없음).

입력
  v1 : /home/ubuntu/shared/datasets/ebnerd/{train,validation_..._half,test}_*.parquet (읽기만)
  h,r: <PRE>/experiments/v2_<run>/candidate_1pos4neg_shuffled/train_sequences_1pos4neg.parquet
       <PRE>/experiments/v2_<run>/split/{validation_sequences_1pos4neg_half,test_sequences_1pos4neg}.parquet
결과: <V2>/features/<run>/ 에 같은 파일 이름으로 (이미 있으면 멈춤)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_test import REPO, load_clicks  # noqa: E402

FILES = ["train_sequences_1pos4neg.parquet", "validation_sequences_1pos4neg_half.parquet", "test_sequences_1pos4neg.parquet"]


def inputs(run: str, pre: Path, exp_prefix: str) -> list:
    if run == "v1":
        d = Path("/home/ubuntu/shared/datasets/ebnerd")
        return [d / f for f in FILES]
    e = pre / "experiments" / f"{exp_prefix}{run}"
    return [e / "candidate_1pos4neg_shuffled" / FILES[0], e / "split" / FILES[1], e / "split" / FILES[2]]


def add_features(src: Path, dst: Path, clicks: dict, pub: dict, window_s: int) -> dict:
    t = pq.read_table(src)
    cand = t.column("candidate_article_ids").combine_chunks()
    lens = pc.list_value_length(cand).to_numpy(zero_copy_only=False)
    if not (lens == 5).all():
        raise ValueError(f"{src.name}: 후보가 5개가 아닌 행이 있습니다")
    aid = pc.list_flatten(cand).to_numpy(zero_copy_only=False).astype(np.int64)
    imp = t.column("impression_time").to_numpy().astype("datetime64[s]").astype(np.int64)
    imp = np.repeat(imp, 5)

    pop = np.zeros(len(aid), dtype=np.int64)
    for i, (a, ti) in enumerate(zip(aid, imp)):
        arr = clicks.get(a)
        if arr is not None:
            pop[i] = np.searchsorted(arr, ti, side="left") - np.searchsorted(arr, ti - window_s, side="left")
    pt = np.array([pub.get(a, np.nan) for a in aid], dtype="float64")
    no_pub = int(np.isnan(pt).sum())
    hours = np.clip((imp - pt) / 3600.0, 0, None)
    logpop = np.log1p(pop).astype(np.float32)
    logage = np.log1p(np.nan_to_num(hours, nan=0.0)).astype(np.float32)

    offsets = pa.array(np.arange(0, len(aid) + 1, 5, dtype=np.int32))
    t = t.append_column("candidate_logpop", pa.ListArray.from_arrays(offsets, pa.array(logpop)))
    t = t.append_column("candidate_logage", pa.ListArray.from_arrays(offsets, pa.array(logage)))
    pq.write_table(t, dst)
    return {"file": dst.name, "rows": t.num_rows, "candidates": int(len(aid)),
            "pop_zero_share": float((pop == 0).mean()), "pop_median": float(np.median(pop)), "pop_max": int(pop.max()),
            "age_hours_median": float(np.nanmedian(hours)), "no_published_time": no_pub}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["v1", "h", "r"])
    ap.add_argument("--pre", default=str(REPO / "data/output/fixedcluster_v1"))
    ap.add_argument("--v2", default=None, help="기본 <pre>/v2")
    ap.add_argument("--raw", default=str(REPO / "data/raw"))
    ap.add_argument("--exp-prefix", default="v2_")
    ap.add_argument("--window-hours", type=float, default=24.0)
    a = ap.parse_args()
    pre = Path(a.pre).expanduser()
    v2 = Path(a.v2).expanduser() if a.v2 else pre / "v2"
    raw = Path(a.raw).expanduser()
    out = v2 / "features" / a.run
    if out.exists():
        sys.exit(f"{out} 가 이미 있습니다 (덮어쓰지 않음)")
    srcs = inputs(a.run, pre, a.exp_prefix)
    for s in srcs:
        if not s.exists():
            sys.exit(f"입력 없음: {s}")
    out.mkdir(parents=True)

    print("클릭 기록·발행 시각 읽는 중 ...", flush=True)
    clicks = load_clicks(raw)
    arts = pd.read_parquet(raw / "articles.parquet", columns=["article_id", "published_time"])
    pub = dict(zip(arts.article_id.astype(np.int64), arts.published_time.to_numpy().astype("datetime64[s]").astype(np.int64)))

    report = {"run": a.run, "window_hours": a.window_hours, "inputs": [str(s) for s in srcs], "files": []}
    for s in srcs:
        r = add_features(s, out / s.name, clicks, pub, int(a.window_hours * 3600))
        report["files"].append(r)
        print(f"{r['file']:<45} 행 {r['rows']:>8,} · 클릭 0인 후보 {r['pop_zero_share']:.1%} · 클릭 중앙값 {r['pop_median']:.0f} "
              f"(최대 {r['pop_max']:,}) · 나이 중앙값 {r['age_hours_median']:.1f}시간 · 발행 시각 없음 {r['no_published_time']}")
    (out / "features_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"== 완료: {out}")


if __name__ == "__main__":
    main()
