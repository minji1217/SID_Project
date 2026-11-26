"""SURGE 아이디어 1·2 결합 (재학습 없음): Transformer 점수에 사건 인기도와 매칭 점수를 더해 Top-1 비교.

    python experiments/v2_yeomin_rerun/surge_fusion.py --runs h r --seed 42

popularity_fusion.py (모델 + λ·인기 + μ·나이) 를 그대로 기준으로 두고, 항을 하나씩 더한다.

  점수 = 모델 + λ·log(1+자기 클릭) + μ·log(1+나이)                       ← 지금 (A)
       + λe·log(1+사건 클릭)                                            ← 아이디어 1 (B)
       + c·매칭                                                         ← 아이디어 2 (C)
       (둘 다 = D)

아이디어 1 · 사건 인기도
  클릭 기록 = train + validation behaviors 의 모든 클릭 (클릭 시각 = 그 노출의 impression_time).
  클릭된 기사에 SID 파일의 (c1, c2) 를 붙이고, (c1, c2) → 정렬된 클릭 시각. SID 없는 클릭은 건너뛴다.
  후보마다 [노출 시각 − 24h, 노출 시각) 안에서 후보와 (c1, c2) 가 같은 클릭 수를 센다.
  (기사 인기도와 사건 인기도의 창 24h 는 --window-hours 로 바꾼다. popularity_window.py 결과 12h 가 validation 최고)

아이디어 2 · 매칭 (비율표 R)
  행마다 그 행의 history 로 w = exp(−(노출 시각 − 클릭 시각)/τ) 를 매기고,
  (c1, c2) 별 w 합 ÷ 전체 w 합 = 비율표 R.
  매칭 = R 에 후보 (c1, c2) 칸이 있으면 그 값, 없으면 ν × (R 에서 c1 이 같은 칸들의 합), 둘 다 없으면 0.
  history 의 (c1, c2) 는 1pos4neg 파일의 history_c1/history_c2 를 쓴다. 기본은 Transformer 입력과 같게
  마지막 50개만 쓴다 (V1 설정 history 50, 여민 data/sequence.py 의 history[-50:]). --max-history 0 이면 전체.
  history 시각은 1pos4neg 에 없어서 복원한다: validation history.parquet 의 (기사, 시각) 과
  validation behaviors 의 현재 기사(그 노출 시각)·클릭(그 노출 시각 직후)으로 (사용자, 기사) 별 시각 목록을 만들고,
  history 의 기사마다 "노출 시각 이전의 가장 최근 시각" 을 쓴다 (같은 기사가 여러 번 있으면 모두 가장 최근 시각).
  시각을 못 찾은 history 항목은 w = 0 으로 두고 그 비율을 출력한다.

가중치 고르기: validation 앞 절반(val_candidate_scores.parquet)에서 Top-1 이 가장 높은 조합, test 에는 한 번만 적용.
  같은 Top-1 이면 가중치가 작은 쪽. D 는 (λ, μ) 를 A 의 최적값 주변 3×3 으로만 찾는다 (격자 크기 때문).

입력
  예측   <V2>/transformer/<run>_seed<seed>/{val,test}_candidate_scores.parquet   (5_predict_val.sh, 4_transformer.sh)
  1pos4neg  v1 : /home/ubuntu/shared/datasets/ebnerd/{validation_sequences_1pos4neg_half,test_sequences_1pos4neg}.parquet
            h,r: <PRE>/experiments/v2_<run>/split/ 같은 파일 이름
  SID    v1 : ~/minji/uni_post_rqvae/article_semantic_ids.parquet,  h,r: <V2>/sid/<run>/article_semantic_ids.parquet
  원본   <raw>/{train,validation}/behaviors.parquet, validation/history.parquet, articles.parquet
결과: <V2>/analysis/surge_fusion_seed<seed>.json, .csv (이미 있으면 멈춤). --max-history 를 바꾸면 이름 끝에 _hist<N>/_histall,
  --split-history 면 E (최근 N개 매칭 + 그 이전 history 매칭, 가중치 따로) 를 더하고 이름 끝에 _split.
  --wide-grid 면 _wide (τ 336h·720h·∞, ν 0.5·0.75·1, c 32·64·128 추가)
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_test import REPO  # noqa: E402
from popularity_fusion import LAMBDAS, MUS, metrics  # noqa: E402

LAMBDA_E = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 4]
C_MATCH = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
TAUS = [24.0, 48.0, 96.0, 168.0]       # 시간
NUS = [0, 0.1, 0.2, 0.3]
# --wide-grid: 지난 결과(τ 168h, ν 0.3, c 16)가 모두 격자 끝이라 바깥쪽 후보를 더한다. τ = inf 는 감쇠 없음 (w = 1).
WIDE = {"TAUS": [336.0, 720.0, float("inf")], "NUS": [0.5, 0.75, 1.0], "C_MATCH": [32, 64, 128]}
WINDOW_H = 24.0
FILES = {"val": "validation_sequences_1pos4neg_half.parquet", "test": "test_sequences_1pos4neg.parquet"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_s(x) -> np.ndarray:
    return np.asarray(x).astype("datetime64[s]").astype(np.int64)


def paths(run: str, pre: Path, v2: Path, exp_prefix: str, seed: str):
    if run == "v1":
        seq = Path("/home/ubuntu/shared/datasets/ebnerd")
        sid = Path("~/minji/uni_post_rqvae/article_semantic_ids.parquet").expanduser()
    else:
        seq = pre / "experiments" / f"{exp_prefix}{run}" / "split"
        sid = v2 / "sid" / run / "article_semantic_ids.parquet"
    tdir = v2 / "transformer" / f"{run}_seed{seed}"
    return {"seq": {k: seq / f for k, f in FILES.items()}, "sid": sid,
            "pred": {"val": tdir / "val_candidate_scores.parquet", "test": tdir / "test_candidate_scores.parquet"}}


# ---------------------------------------------------------------- 공통 재료 (run 과 무관)
def load_click_log(raw: Path) -> pd.DataFrame:
    parts = []
    for split in ["train", "validation"]:
        b = pd.read_parquet(raw / split / "behaviors.parquet", columns=["impression_time", "article_ids_clicked"])
        b = b.explode("article_ids_clicked").dropna(subset=["article_ids_clicked"])
        parts.append(pd.DataFrame({"a": b.article_ids_clicked.astype(np.int64).to_numpy(), "t": to_s(b.impression_time)}))
    return pd.concat(parts, ignore_index=True)


def sorted_times_by(keys: np.ndarray, t: np.ndarray) -> dict:
    o = np.lexsort((t, keys))
    k, t = keys[o], t[o]
    cut = np.flatnonzero(np.diff(k)) + 1
    return {int(kk): tt for kk, tt in zip(k[np.r_[0, cut]], np.split(t, cut))}


def count_window(d: dict, keys: np.ndarray, t: np.ndarray, window_s: int) -> np.ndarray:
    out = np.zeros(len(keys), dtype=np.int64)
    for i, (k, ti) in enumerate(zip(keys, t)):
        arr = d.get(int(k))
        if arr is not None:
            out[i] = np.searchsorted(arr, ti, side="left") - np.searchsorted(arr, ti - window_s, side="left")
    return out


class HistoryTimes:
    """(사용자, 기사) → 그 사용자가 그 기사를 history 에 넣은 시각들. query 는 노출 시각 이전의 가장 최근 시각."""

    def __init__(self, raw: Path):
        h = pd.read_parquet(raw / "validation" / "history.parquet", columns=["user_id", "article_id_fixed", "impression_time_fixed"])
        h = h.dropna(subset=["user_id", "article_id_fixed", "impression_time_fixed"])
        h = h.explode(["article_id_fixed", "impression_time_fixed"]).dropna()
        u = [h.user_id.astype(np.int64).to_numpy()]
        a = [h.article_id_fixed.astype(np.int64).to_numpy()]
        t = [to_s(pd.to_datetime(h.impression_time_fixed))]
        b = pd.read_parquet(raw / "validation" / "behaviors.parquet", columns=["user_id", "impression_time", "article_id", "article_ids_clicked"])
        cur = b.dropna(subset=["article_id"])
        u.append(cur.user_id.astype(np.int64).to_numpy()); a.append(cur.article_id.astype(np.int64).to_numpy()); t.append(to_s(cur.impression_time))
        clk = b.explode("article_ids_clicked").dropna(subset=["article_ids_clicked"])
        # 클릭은 그 노출의 sample 을 만든 뒤에 history 에 들어가므로, 노출 시각 + 1초로 둔다 (같은 노출에는 안 보임)
        u.append(clk.user_id.astype(np.int64).to_numpy()); a.append(clk.article_ids_clicked.astype(np.int64).to_numpy()); t.append(to_s(clk.impression_time) + 1)
        u, a, t = np.concatenate(u), np.concatenate(a), np.concatenate(t)
        self.t0 = int(t.min()) - 1
        pair = u * 100_000_000 + a                                  # article_id < 1e8
        self.uniq, idx = np.unique(pair, return_inverse=True)
        comp = idx.astype(np.int64) * (1 << 32) + (t - self.t0)
        self.comp = np.sort(comp)

    def query(self, user: np.ndarray, art: np.ndarray, T: np.ndarray) -> np.ndarray:
        """시각(초) 배열, 못 찾으면 -1."""
        pair = user * 100_000_000 + art
        pos = np.searchsorted(self.uniq, pair)
        pos_c = np.minimum(pos, len(self.uniq) - 1)
        ok = self.uniq[pos_c] == pair
        q = pos_c.astype(np.int64) * (1 << 32) + (T - self.t0)
        j = np.searchsorted(self.comp, q, side="right") - 1
        j_c = np.maximum(j, 0)
        ok &= (j >= 0) & ((self.comp[j_c] >> 32) == pos_c)
        out = np.where(ok, (self.comp[j_c] & ((1 << 32) - 1)) + self.t0, -1)
        return out


# ---------------------------------------------------------------- run 별
def load_split(pred_path: Path, seq_path: Path) -> tuple:
    p = pd.read_parquet(pred_path, columns=["sample_index", "impression_id", "user_id", "impression_time", "article_id", "c1", "c2", "label", "candidate_score"])
    p = p.sort_values(["sample_index"], kind="stable").reset_index(drop=True)
    n = p.sample_index.nunique()
    if len(p) != 5 * n:
        raise ValueError(f"{pred_path.name}: 행마다 후보 5개가 아닙니다")
    seq = pq.read_table(seq_path, columns=["impression_id", "user_id", "impression_time", "candidate_article_ids",
                                           "history_article_ids", "history_c1", "history_c2"])
    if seq.num_rows != n or p.sample_index.max() != n - 1:
        raise ValueError(f"{seq_path.name} 행 수 {seq.num_rows} ≠ 예측 sample 수 {n}")
    # sample_index = 1pos4neg 행 번호 인지 확인 (predict_sid.py 는 shuffle=False)
    cand = pc.list_flatten(seq.column("candidate_article_ids").combine_chunks()).to_numpy(zero_copy_only=False).astype(np.int64)
    if not np.array_equal(cand, p.article_id.astype(np.int64).to_numpy()):
        raise ValueError(f"{pred_path.name}: 후보 기사 순서가 {seq_path.name} 과 다릅니다 (sample_index 정렬 확인)")
    imp = to_s(seq.column("impression_time").to_numpy())
    if not np.array_equal(np.repeat(imp, 5), to_s(pd.to_datetime(p.impression_time))):
        raise ValueError("노출 시각이 다릅니다")
    p["article_id"] = p.article_id.astype(np.int64)
    p["label"] = p.label.astype(int)
    p["t"] = np.repeat(imp, 5)
    return p, seq


def match_features(p: pd.DataFrame, seq, ht: HistoryTimes, taus: list, max_history: int = 50, older_only: bool = False) -> dict:
    """τ 마다 후보별 (자기 칸 값, 자기 칸 있음, 같은 c1 칸 합). 시각 못 찾은 비율도 반환.
    max_history > 0 이면 행마다 history 의 마지막 max_history 개만 쓴다 (Transformer 입력과 같게).
    older_only 이면 반대로 마지막 max_history 개를 뺀 나머지 (그보다 이전 클릭) 만 쓴다."""
    lens = pc.list_value_length(seq.column("history_c1").combine_chunks()).to_numpy(zero_copy_only=False).astype(np.int64)
    row = np.repeat(np.arange(seq.num_rows), lens)
    hc1 = pc.list_flatten(seq.column("history_c1").combine_chunks()).to_numpy(zero_copy_only=False).astype(np.int64)
    hc2 = pc.list_flatten(seq.column("history_c2").combine_chunks()).to_numpy(zero_copy_only=False).astype(np.int64)
    ha = pc.list_flatten(seq.column("history_article_ids").combine_chunks()).to_numpy(zero_copy_only=False).astype(np.int64)
    if not (len(hc1) == len(hc2) == len(ha)):
        raise ValueError("history_c1 / history_c2 / history_article_ids 길이가 다릅니다")
    if max_history > 0:
        start = np.repeat(np.cumsum(lens) - lens, lens)            # 행의 첫 원소 위치
        from_end = np.repeat(lens, lens) - (np.arange(len(row)) - start)   # 뒤에서 몇 번째 (1 = 마지막)
        keep = (from_end > max_history) if older_only else (from_end <= max_history)
        row, hc1, hc2, ha = row[keep], hc1[keep], hc2[keep], ha[keep]
        lens = np.maximum(lens - max_history, 0) if older_only else np.minimum(lens, max_history)
    user = seq.column("user_id").to_numpy().astype(np.int64)[row]
    T = to_s(seq.column("impression_time").to_numpy())[row]
    ht_s = ht.query(user, ha, T)
    found = ht_s >= 0
    hours = np.where(found, (T - ht_s) / 3600.0, np.inf)
    ev = hc1 * 100_000 + hc2
    crow = np.repeat(np.arange(seq.num_rows), 5)
    cev = p.c1.to_numpy().astype(np.int64) * 100_000 + p.c2.to_numpy().astype(np.int64)
    cc1 = p.c1.to_numpy().astype(np.int64)
    out = {"history_items": int(len(row)), "history_time_missing": float(1 - found.mean()) if len(row) else 0.0,
           "rows_without_history": int((lens == 0).sum())}
    for tau in taus:
        with np.errstate(invalid="ignore"):
            w = np.where(found, np.exp(-hours / tau), 0.0)   # τ = inf → 시각 찾은 항목은 w = 1
        h = pd.DataFrame({"row": row, "ev": ev, "c1": hc1, "w": w})
        tot = h.groupby("row").w.sum()
        cell = h.groupby(["row", "ev"]).w.sum()
        c1sum = h.groupby(["row", "c1"]).w.sum()
        denom = tot.reindex(crow).to_numpy()
        own_w = cell.reindex(pd.MultiIndex.from_arrays([crow, cev])).to_numpy()
        c1_w = c1sum.reindex(pd.MultiIndex.from_arrays([crow, cc1])).to_numpy()
        has = ~np.isnan(own_w)
        with np.errstate(invalid="ignore", divide="ignore"):
            own = np.where(has & (denom > 0), own_w / denom, 0.0)
            c1s = np.where(~np.isnan(c1_w) & (denom > 0), c1_w / denom, 0.0)
        out[tau] = (own, has, c1s)
    return out


def matching(f: tuple, nu: float) -> np.ndarray:
    own, has, c1s = f
    return np.where(has, own, nu * c1s)


def top1_fast(S: np.ndarray, Y: np.ndarray) -> float:
    S = S.reshape(-1, 5); Y = Y.reshape(-1, 5)
    top = S == S.max(axis=1, keepdims=True)
    return float(((top & (Y == 1)).sum(1) / top.sum(1)).mean())


def pick(grid: dict):
    return max(grid, key=lambda k: (round(grid[k], 10), [-abs(x) for x in k]))


def edge_warn(name: str, best: tuple, lists: list) -> list:
    """고른 값이 격자 맨 끝이면 알림 (격자를 넓혀 다시 볼 것)."""
    return [f"{name}: {v} 가 격자 끝 ({L[0]}..{L[-1]})" for v, L in zip(best, lists) if len(L) > 1 and v in (L[0], L[-1]) and v != 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["v1", "h", "r"])
    ap.add_argument("--seed", default="42")
    ap.add_argument("--pre", default=str(REPO / "data/output/fixedcluster_v1"))
    ap.add_argument("--v2", default=None, help="기본 <pre>/v2")
    ap.add_argument("--raw", default=str(REPO / "data/raw"))
    ap.add_argument("--exp-prefix", default="v2_")
    ap.add_argument("--max-history", type=int, default=50, help="매칭에 쓸 history 마지막 N개 (Transformer V1 = 50). 0 = 전체")
    ap.add_argument("--window-hours", type=float, default=WINDOW_H, help="기사 · 사건 인기도를 셀 직전 시간 (기본 24)")
    ap.add_argument("--split-history", action="store_true",
                    help="E 추가: 최근 --max-history 개 매칭과 그 이전 history 매칭을 따로 더함 (가중치 따로)")
    ap.add_argument("--wide-grid", action="store_true", help="τ, ν, c 후보를 바깥쪽으로 넓힘 (기본 격자는 그대로 포함)")
    a = ap.parse_args()
    if a.split_history and a.max_history <= 0:
        sys.exit("--split-history 는 --max-history > 0 일 때만 (최근 N개 / 그 이전으로 나눔)")
    global TAUS, NUS, C_MATCH
    if a.wide_grid:
        TAUS, NUS, C_MATCH = TAUS + WIDE["TAUS"], NUS + WIDE["NUS"], C_MATCH + WIDE["C_MATCH"]
    pre = Path(a.pre).expanduser()
    v2 = Path(a.v2).expanduser() if a.v2 else pre / "v2"
    raw = Path(a.raw).expanduser()
    out_dir = v2 / "analysis"
    tag = f"seed{a.seed}" + ("" if a.max_history == 50 else f"_hist{a.max_history or 'all'}") + ("_wide" if a.wide_grid else "") + ("" if a.window_hours == WINDOW_H else f"_w{a.window_hours:g}") + ("_split" if a.split_history else "")
    jpath, cpath = out_dir / f"surge_fusion_{tag}.json", out_dir / f"surge_fusion_{tag}.csv"
    if jpath.exists() or cpath.exists():
        sys.exit(f"멈춤: {jpath} 가 이미 있습니다 (덮어쓰지 않음)")
    window_s = int(a.window_hours * 3600)

    log("클릭 기록 읽는 중")
    clicks = load_click_log(raw)
    by_art = sorted_times_by(clicks.a.to_numpy(), clicks.t.to_numpy())
    arts = pd.read_parquet(raw / "articles.parquet", columns=["article_id", "published_time"])
    pub = dict(zip(arts.article_id.astype(np.int64), to_s(arts.published_time)))
    log("history 시각 목록 만드는 중")
    ht = HistoryTimes(raw)

    results, rows = [], []
    for run in a.runs:
        P = paths(run, pre, v2, a.exp_prefix, a.seed)
        need = [P["sid"], *P["seq"].values(), *P["pred"].values()]
        miss = [str(x) for x in need if not x.exists()]
        if miss:
            log(f"[{run}] 입력 없음 → 건너뜀: {miss}")
            continue
        log(f"[{run}] SID·사건 클릭 준비")
        sid = pd.read_parquet(P["sid"], columns=["article_id", "c1", "c2"])
        a2e = dict(zip(sid.article_id.astype(np.int64), sid.c1.astype(np.int64) * 100_000 + sid.c2.astype(np.int64)))
        cev = clicks.a.map(a2e)
        skipped = float(cev.isna().mean())
        ok = cev.notna().to_numpy()
        by_ev = sorted_times_by(cev[ok].astype(np.int64).to_numpy(), clicks.t.to_numpy()[ok])

        D, OLD, info = {}, {}, {"clicks_without_sid": skipped}
        for split in ["val", "test"]:
            log(f"[{run}] {split} 읽기·특징 계산")
            p, seq = load_split(P["pred"][split], P["seq"][split])
            aid, t = p.article_id.to_numpy(), p.t.to_numpy()
            p["logpop"] = np.log1p(count_window(by_art, aid, t, window_s))
            pt = np.array([pub.get(x, np.nan) for x in aid], dtype="float64")
            p["logage"] = np.log1p(np.nan_to_num(np.clip((t - pt) / 3600.0, 0, None), nan=0.0))
            ev = p.c1.to_numpy().astype(np.int64) * 100_000 + p.c2.to_numpy().astype(np.int64)
            p["logepop"] = np.log1p(count_window(by_ev, ev, t, window_s))
            mf = match_features(p, seq, ht, TAUS, a.max_history)
            info[split] = {k: mf[k] for k in ("history_items", "history_time_missing", "rows_without_history")}
            pos = p.label.to_numpy() == 1
            f48 = mf[48.0]
            info[split]["정답 후보 사건이 R 에 있음"] = float(f48[1][pos].mean())
            info[split]["오답 후보 사건이 R 에 있음"] = float(f48[1][~pos].mean())
            info[split]["정답 자기 클릭 < 10"] = float((np.expm1(p.logpop.to_numpy()[pos]) < 10).mean())
            D[split] = (p, mf)
            if a.split_history:
                OLD[split] = match_features(p, seq, ht, TAUS, a.max_history, older_only=True)
                info[split]["이전 history 항목"] = OLD[split]["history_items"]
                info[split]["이전 history 없는 행"] = OLD[split]["rows_without_history"]

        (pv, mv), (pt_, mt) = D["val"], D["test"]
        Yv, Yt = pv.label.to_numpy(), pt_.label.to_numpy()
        sv, st = pv.candidate_score.to_numpy(), pt_.candidate_score.to_numpy()
        V = {k: pv[k].to_numpy() for k in ("logpop", "logage", "logepop")}
        Tt = {k: pt_[k].to_numpy() for k in ("logpop", "logage", "logepop")}

        log(f"[{run}] 가중치 고르기 (validation)")
        gA = {(l, m): top1_fast(sv + l * V["logpop"] + m * V["logage"], Yv) for l in LAMBDAS for m in MUS}
        bA = pick(gA)
        gB = {(l, m, le): top1_fast(sv + l * V["logpop"] + m * V["logage"] + le * V["logepop"], Yv)
              for l in LAMBDAS for m in MUS for le in LAMBDA_E}
        bB = pick(gB)
        Mv = {(tau, nu): matching(mv[tau], nu) for tau in TAUS for nu in NUS}
        Mt = {(tau, nu): matching(mt[tau], nu) for tau in TAUS for nu in NUS}
        gC = {}
        for l, m in itertools.product(LAMBDAS, MUS):
            base = sv + l * V["logpop"] + m * V["logage"]
            for (tau, nu), M in Mv.items():
                for c in C_MATCH:
                    gC[(l, m, tau, nu, c)] = top1_fast(base + c * M, Yv)
        bC = pick(gC)
        near = lambda L, x: L[max(0, L.index(x) - 1): L.index(x) + 2]
        gD = {}
        for l, m in itertools.product(near(LAMBDAS, bA[0]), near(MUS, bA[1])):
            base = sv + l * V["logpop"] + m * V["logage"]
            for le in LAMBDA_E:
                b2 = base + le * V["logepop"]
                for (tau, nu), M in Mv.items():
                    for c in C_MATCH:
                        gD[(l, m, le, tau, nu, c)] = top1_fast(b2 + c * M, Yv)
        bD = pick(gD)
        if a.split_history:   # E: C 의 λ, μ 를 두고 최근 N개 매칭 (c1) + 그 이전 history 매칭 (c2) 을 따로 고른다
            Ov = {(tau, nu): matching(OLD["val"][tau], nu) for tau in TAUS for nu in NUS}
            Ot = {(tau, nu): matching(OLD["test"][tau], nu) for tau in TAUS for nu in NUS}
            l, m = bC[0], bC[1]
            base = sv + l * V["logpop"] + m * V["logage"]
            gE = {}
            for (tau, nu), M in Mv.items():
                for c1 in C_MATCH:
                    b1 = base + c1 * M
                    for c2 in C_MATCH:
                        gE[(l, m, tau, nu, c1, c2)] = top1_fast(b1 + c2 * Ov[(tau, nu)], Yv)
            bE = pick(gE)

        l, m = bA
        test = {
            "모델만": metrics(pt_, st),
            f"A 지금 (λ={l}, μ={m})": metrics(pt_, st + l * Tt["logpop"] + m * Tt["logage"]),
        }
        l, m, le = bB
        test[f"B +아이디어1 (λ={l}, μ={m}, λe={le})"] = metrics(pt_, st + l * Tt["logpop"] + m * Tt["logage"] + le * Tt["logepop"])
        l, m, tau, nu, c = bC
        test[f"C +아이디어2 (λ={l}, μ={m}, τ={tau:g}h, ν={nu}, c={c})"] = metrics(pt_, st + l * Tt["logpop"] + m * Tt["logage"] + c * Mt[(tau, nu)])
        l, m, le, tau, nu, c = bD
        test[f"D +둘 다 (λ={l}, μ={m}, λe={le}, τ={tau:g}h, ν={nu}, c={c})"] = metrics(
            pt_, st + l * Tt["logpop"] + m * Tt["logage"] + le * Tt["logepop"] + c * Mt[(tau, nu)])
        # 참고: 각 항 하나만
        if a.split_history:
            l, m, tau, nu, c1, c2 = bE
            test[f"E 최근{a.max_history}+이전 매칭 (λ={l}, μ={m}, τ={tau:g}h, ν={nu}, c최근={c1}, c이전={c2})"] = metrics(
                pt_, st + l * Tt["logpop"] + m * Tt["logage"] + c1 * Mt[(tau, nu)] + c2 * Ot[(tau, nu)])
        test["참고: 사건 인기도만"] = metrics(pt_, Tt["logepop"])
        test["참고: 매칭만 (τ=48h, ν=0.2)"] = metrics(pt_, Mt[(48.0, 0.2)])

        warn = (edge_warn("A", bA, [LAMBDAS, MUS]) + edge_warn("B", bB, [LAMBDAS, MUS, LAMBDA_E])
                + edge_warn("C", bC, [LAMBDAS, MUS, TAUS, NUS, C_MATCH]) + edge_warn("D", bD[2:], [LAMBDA_E, TAUS, NUS, C_MATCH])
                + (edge_warn("E", bE[2:], [TAUS, NUS, C_MATCH, C_MATCH]) if a.split_history else []))
        r = {"run": run, "seed": a.seed, "max_history": a.max_history, "wide_grid": a.wide_grid, "window_hours": a.window_hours,
             "grid": {"TAUS": TAUS, "NUS": NUS, "C_MATCH": C_MATCH}, "info": info, "grid_edge_warnings": warn,
             "val_top1": {"A": gA[bA], "B": gB[bB], "C": gC[bC], "D": gD[bD], **({"E": gE[bE]} if a.split_history else {})},
             "best": {"A": bA, "B": bB, "C": bC, "D": bD, **({"E": bE} if a.split_history else {})}, "split_history": a.split_history, "test": test}
        results.append(r)
        print(f"\n===== {run.upper()} (seed {a.seed}, 매칭 history {'전체' if a.max_history == 0 else f'마지막 {a.max_history}개'}, 인기도 창 {a.window_hours:g}시간) =====")
        print(f"SID 없어 건너뛴 클릭 {skipped:.2%} · " + " · ".join(
            f"{s}: history 시각 못 찾음 {info[s]['history_time_missing']:.2%}, 후보 사건이 R 에 있음 정답 {info[s]['정답 후보 사건이 R 에 있음']:.1%} / 오답 {info[s]['오답 후보 사건이 R 에 있음']:.1%}"
            for s in ("val", "test")))
        for w_ in warn:
            print("  주의 " + w_)
        print("validation Top-1: " + " · ".join(f"{k} {v:.2%}" for k, v in r["val_top1"].items()))
        for name, mm in test.items():
            print(f"  test {name:<58} Top-1 {mm['top1']:.2%} · AUC {mm['auc']:.4f} · MRR {mm['mrr']:.4f} · nDCG@5 {mm['ndcg@5']:.4f}")
            rows.append({"run": run, "seed": a.seed, "방식": name, **mm})

    if results:
        out_dir.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        pd.DataFrame(rows).to_csv(cpath, index=False)
        log(f"저장: {jpath}, {cpath}")


if __name__ == "__main__":
    main()
