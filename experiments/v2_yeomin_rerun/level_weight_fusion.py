"""자리별(level) 가중치 + 인기·나이 결합 (재학습 없음). v2 조건(섞은 후보 · fp32)에서 다시 확인한다.

    python experiments/v2_yeomin_rerun/level_weight_fusion.py                 # V1 · H · R × seed 42 · 123 · 2026 중 있는 것
    python experiments/v2_yeomin_rerun/level_weight_fusion.py --runs h --seeds 42

점수
  모델            S = L1 + L2 + L3                         (L = 여민 predict_sid.py가 저장한 c1/c2/c3 log 확률. 지금 Transformer 점수)
  level 가중치     S = L1 + a2·L2 + a3·L3                   (a1 = 1 고정, a2 · a3 는 아래 격자)
  결합 + 나이      S = L1 + L2 + L3 + λ·logpop + μ·logage   (popularity_fusion.py --with-age 와 같은 격자 · 같은 규칙)
  level + 결합     S = L1 + a2·L2 + a3·L3 + λ·logpop + μ·logage  (a2 · a3 · λ · μ 를 같이 고름)
값 고르기: validation 앞 절반 Top-1 최대 (같으면 기본값에 가까운 쪽: |a2−1|+|a3−1| 작은 쪽, 그다음 |λ|, |μ| 작은 쪽). test 에는 한 번만.
지표: popularity_fusion.metrics 와 같음 (동점 기대값).

예전 결과(10/08, 예전 V1 데이터 = 후보 안 섞음 · AMP): (1, 0.5, 0.1) 이 27.07% → 29.46%. 그 데이터는 동점이 정답으로 처리돼
부풀려져 있었으므로 지금 조건에서 다시 본다.
입력: <V2>/transformer/<run>_seed<S>/{val,test}_candidate_scores.parquet, data/raw (클릭 · 발행 시각) — 읽기만
결과: 화면 + <V2>/analysis/level_weight_fusion.csv (이미 있으면 _2, _3 …)
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_test import REPO, load_clicks  # noqa: E402
from popularity_fusion import LAMBDAS, MUS, add_age, add_pop, metrics  # noqa: E402

A2 = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5]
A3 = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
COLS = ["sample_index", "impression_time", "article_id", "label", "c1_log_prob", "c2_log_prob", "c3_log_prob"]


def load(path: Path, clicks: dict, pub: dict, window_s: int) -> pd.DataFrame:
    p = pd.read_parquet(path, columns=COLS)
    p["article_id"] = p.article_id.astype(np.int64)
    p["label"] = p.label.astype(int)
    p = add_pop(p, clicks, window_s)
    p = add_age(p, pub).sort_values("sample_index", kind="stable").reset_index(drop=True)
    assert (p.groupby("sample_index").size() == 5).all(), "행마다 후보 5개여야 합니다"
    return p


def top1_fast(s: np.ndarray, y: np.ndarray) -> float:
    pos = (s * y).sum(1, keepdims=True)
    neg = y == 0
    g = ((s > pos) & neg).sum(1)
    t = ((s == pos) & neg).sum(1)
    return float(np.where(g == 0, 1.0 / (1 + t), 0.0).mean())


def parts(p: pd.DataFrame):
    r = lambda c: p[c].to_numpy(dtype=np.float64).reshape(-1, 5)
    return r("c1_log_prob"), r("c2_log_prob"), r("c3_log_prob"), r("logpop"), r("logage"), p.label.to_numpy().reshape(-1, 5)


def run_one(val: pd.DataFrame, test: pd.DataFrame) -> dict:
    v1, v2, v3, vp, va, vy = parts(val)
    t1, t2, t3, tp, ta, ty = parts(test)
    prior = lambda a2, a3: abs(a2 - 1) + abs(a3 - 1)

    # level 가중치만
    g = {(a2, a3): top1_fast(v1 + a2 * v2 + a3 * v3, vy) for a2, a3 in itertools.product(A2, A3)}
    lw = max(g, key=lambda k: (g[k], -prior(*k)))
    # 결합 + 나이 (level 1, 1, 1)
    g2 = {(l, m): top1_fast(v1 + v2 + v3 + l * vp + m * va, vy) for l, m in itertools.product(LAMBDAS, MUS)}
    fm = max(g2, key=lambda k: (g2[k], -abs(k[0]), -abs(k[1])))
    # level + 결합 + 나이
    base = {(a2, a3): v1 + a2 * v2 + a3 * v3 for a2, a3 in itertools.product(A2, A3)}
    g3 = {}
    for (a2, a3), s in base.items():
        for l, m in itertools.product(LAMBDAS, MUS):
            g3[(a2, a3, l, m)] = top1_fast(s + l * vp + m * va, vy)
    al = max(g3, key=lambda k: (g3[k], -prior(k[0], k[1]), -abs(k[2]), -abs(k[3])))

    ev = lambda s: metrics(test, s.reshape(-1))
    res = {
        "모델 (1,1,1)": ev(t1 + t2 + t3),
        "c1만": ev(t1), "c1+c2": ev(t1 + t2),
        f"level ({1},{lw[0]},{lw[1]})": ev(t1 + lw[0] * t2 + lw[1] * t3),
        "인기만": ev(tp),
        f"결합+나이 (λ={fm[0]}, μ={fm[1]})": ev(t1 + t2 + t3 + fm[0] * tp + fm[1] * ta),
        f"level+결합+나이 (1,{al[0]},{al[1]}, λ={al[2]}, μ={al[3]})": ev(t1 + al[0] * t2 + al[1] * t3 + al[2] * tp + al[3] * ta),
    }
    chosen = {"level": lw, "level_val_top1": g[lw], "level111_val_top1": g[(1, 1)],
              "fusion": fm, "fusion_val_top1": g2[fm], "all": al, "all_val_top1": g3[al]}
    return {"test": res, "chosen": chosen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default=str(REPO / "data/output/fixedcluster_v1"))
    ap.add_argument("--v2", default=None, help="기본 <pre>/v2")
    ap.add_argument("--raw", default=str(REPO / "data/raw"))
    ap.add_argument("--runs", nargs="+", default=["v1", "h", "r"])
    ap.add_argument("--seeds", nargs="+", default=["42", "123", "2026"])
    ap.add_argument("--window-hours", type=float, default=24.0)
    a = ap.parse_args()
    v2 = Path(a.v2).expanduser() if a.v2 else Path(a.pre).expanduser() / "v2"
    raw = Path(a.raw).expanduser()

    print("클릭 기록 · 발행 시각 읽는 중 ...", flush=True)
    clicks = load_clicks(raw)
    arts = pd.read_parquet(raw / "articles.parquet", columns=["article_id", "published_time"])
    pub = dict(zip(arts.article_id.astype(np.int64), arts.published_time.to_numpy().astype("datetime64[s]").astype(np.int64)))

    rows = []
    for run in a.runs:
        for seed in a.seeds:
            d = v2 / "transformer" / f"{run}_seed{seed}"
            vf, tf = d / "val_candidate_scores.parquet", d / "test_candidate_scores.parquet"
            if not (vf.exists() and tf.exists()):
                continue
            ws = int(a.window_hours * 3600)
            r = run_one(load(vf, clicks, pub, ws), load(tf, clicks, pub, ws))
            c = r["chosen"]
            print(f"\n===== {run.upper()} seed {seed} =====")
            print(f"validation Top-1: (1,1,1) {c['level111_val_top1']:.2%} → level {c['level']} {c['level_val_top1']:.2%} · "
                  f"결합+나이 {c['fusion']} {c['fusion_val_top1']:.2%} · level+결합+나이 {c['all']} {c['all_val_top1']:.2%}")
            for name, m in r["test"].items():
                print(f"  test {name:<44} Top-1 {m['top1']:.2%} · AUC {m['auc']:.4f} · MRR {m['mrr']:.4f} · nDCG@5 {m['ndcg@5']:.4f}")
                kind = name.split(" (")[0]
                rows.append({"실험": run.upper(), "seed": seed, "방식": kind, "설정": name, **m})
    if not rows:
        sys.exit("val/test_candidate_scores.parquet 를 찾지 못했습니다")
    t = pd.DataFrame(rows)
    m = t.groupby(["실험", "방식"], sort=False)[["top1", "auc", "mrr", "ndcg@5"]].mean()
    print("\n== seed 평균 (test)")
    print(m.assign(top1=lambda x: (x.top1 * 100).round(2)).round(4).to_string())
    out = v2 / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    f, k = out / "level_weight_fusion.csv", 2
    while f.exists():
        f, k = out / f"level_weight_fusion_{k}.csv", k + 1
    t.to_csv(f, index=False)
    print(f"\n저장: {f}")


if __name__ == "__main__":
    main()
