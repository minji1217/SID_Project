"""SID semantic consistency 분석 — pair 생성 + similarity 계산.

입력 파일은 읽기만 한다. seed 42 고정.
"""
from __future__ import annotations
import json, re, itertools, unicodedata
from math import comb
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42
U = "/root/.claude/uploads/f5109944-18fd-59ef-9d42-2e3b01694dcf"
LARGE_CAP = 50_000

# ---------------------------------------------------------------- 토큰화
PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
WS = re.compile(r"\s+")

def tokenize(text: str) -> list[str]:
    """lowercase -> punctuation 제거 -> whitespace 분할 -> 빈 token 제거.

    덴마크어 원문을 번역하거나 어간 추출하지 않는다. æøå는 \\w에 포함된다.
    """
    if not isinstance(text, str):
        return []
    t = unicodedata.normalize("NFC", text).lower()
    t = PUNCT.sub(" ", t)
    return [tok for tok in WS.split(t) if tok]

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")      # 둘 다 비면 0이 아니라 결측
    u = a | b
    return len(a & b) / len(u) if u else float("nan")

# ---------------------------------------------------------------- 데이터
def load() -> pd.DataFrame:
    sid = pd.read_parquet(f"{U}/57f50fe6-article_semantic_ids.parquet")
    art = pd.read_parquet(f"{U}/f0158e4f-articles.parquet")
    sid["aid"] = sid.article_id.astype(str).str.strip()
    art["aid"] = art.article_id.astype(str).str.strip()
    cols = ["aid", "title", "subtitle", "body", "category_str"]
    m = sid.merge(art[cols], on="aid", how="left", validate="1:1")
    if m[["title", "body"]].isna().any().any():
        raise ValueError("join 후 title/body에 NaN이 있습니다.")
    return m

def exact_counts(d: pd.DataFrame) -> dict:
    """포함배제로 조건별 전체 pair 수를 정확히 센다."""
    P = lambda s: int(sum(comb(int(n), 2) for n in s if n >= 2))
    total = comb(len(d), 2)
    A, B, C = (P(d.groupby(c).size()) for c in ("c1", "c2", "c3"))
    AB = P(d.groupby(["c1", "c2"]).size())
    AC = P(d.groupby(["c1", "c3"]).size())
    BC = P(d.groupby(["c2", "c3"]).size())
    ABC = P(d.groupby(["c1", "c2", "c3"]).size())
    return {
        "n_articles": len(d), "total_pairs": total,
        "prefix_3_same": ABC,
        "prefix_2_same": AB - ABC,
        "semantic_different": total - (A + B + C) + (AB + AC + BC) - ABC,
    }

# ---------------------------------------------------------------- 샘플러
class PairSet:
    """(min,max) 정규화로 A-B == B-A 중복을 막는다."""
    def __init__(self):
        self.seen: set[tuple[int, int]] = set()
        self.rows: list[dict] = []

    def add(self, i: int, j: int, **meta) -> bool:
        key = (i, j) if i < j else (j, i)
        if i == j or key in self.seen:
            return False
        self.seen.add(key)
        self.rows.append({"idx_a": key[0], "idx_b": key[1], **meta})
        return True

    def __len__(self): return len(self.rows)


def all_prefix3_pairs(d: pd.DataFrame) -> PairSet:
    """(c1,c2,c3)가 같은 모든 pair를 전부 enumerate."""
    ps = PairSet()
    for key, idx in d.groupby(["c1", "c2", "c3"]).indices.items():
        if len(idx) < 2:
            continue
        for a, b in itertools.combinations(sorted(idx), 2):
            ps.add(int(a), int(b), source_group="-".join(map(str, key)))
    return ps


def prefix2_group_balanced(d: pd.DataFrame, n: int, rng) -> PairSet:
    """(c1,c2) group을 round-robin으로 돌면서
    서로 다른 c3 subgroup 2개를 고르고 각각에서 article 하나씩."""
    groups = {}
    for key, idx in d.groupby(["c1", "c2"]).indices.items():
        sub = d.iloc[idx].groupby("c3").indices       # c3 -> 그룹 내 위치
        subs = {c3: idx[v] for c3, v in sub.items()}
        if len(subs) >= 2:
            groups[key] = subs
    keys = sorted(groups)
    ps, stale = PairSet(), 0
    while len(ps) < n and stale < len(keys) * 50:
        rng.shuffle(keys)
        progressed = False
        for key in keys:
            if len(ps) >= n:
                break
            subs = groups[key]
            c3a, c3b = rng.choice(list(subs), size=2, replace=False)
            a = int(rng.choice(subs[c3a])); b = int(rng.choice(subs[c3b]))
            if ps.add(a, b, source_group="-".join(map(str, key)),
                      sub_group_a=int(c3a), sub_group_b=int(c3b)):
                progressed = True
        stale = 0 if progressed else stale + 1
    return ps


def semantic_different_group_balanced(d: pd.DataFrame, n: int, rng) -> PairSet:
    """(c1,c2,c3) group을 균등하게 돌면서 c1,c2,c3가 모두 다른
    compatible group을 골라 각각에서 article 하나씩."""
    gi = {k: v for k, v in d.groupby(["c1", "c2", "c3"]).indices.items()}
    keys = sorted(gi)
    arr = np.array(keys)                      # keys와 같은 순서로 고정
    order = np.arange(len(keys))
    ps, stale = PairSet(), 0
    while len(ps) < n and stale < 2000:
        rng.shuffle(order)
        progressed = False
        for pos in order:
            if len(ps) >= n:
                break
            key = keys[pos]
            ok = ((arr[:, 0] != key[0]) & (arr[:, 1] != key[1])
                  & (arr[:, 2] != key[2]))
            cand = np.flatnonzero(ok)
            if cand.size == 0:
                continue
            other = keys[int(rng.choice(cand))]
            a = int(rng.choice(gi[key]))
            b = int(rng.choice(gi[other]))
            if ps.add(a, b, source_group="-".join(map(str, key)),
                      source_group_b="-".join(map(str, other))):
                progressed = True
        stale = 0 if progressed else stale + 1
    return ps


def prefix2_uniform(d: pd.DataFrame, n: int, rng) -> PairSet:
    """전체 prefix_2_same pair를 enumerate한 뒤 uniform random 추출."""
    pool = []
    for key, idx in d.groupby(["c1", "c2"]).indices.items():
        c3 = d.c3.to_numpy()[idx]
        for a, b in itertools.combinations(range(len(idx)), 2):
            if c3[a] != c3[b]:
                pool.append((int(idx[a]), int(idx[b]), key))
    take = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    ps = PairSet()
    for t in take:
        a, b, key = pool[t]
        ps.add(a, b, source_group="-".join(map(str, key)))
    return ps


def semantic_different_uniform(d: pd.DataFrame, n: int, rng) -> PairSet:
    """article을 uniform하게 두 개 뽑고 조건을 만족할 때만 채택 (rejection)."""
    c = d[["c1", "c2", "c3"]].to_numpy()
    N = len(d)
    ps = PairSet()
    while len(ps) < n:
        a = rng.integers(0, N, size=n); b = rng.integers(0, N, size=n)
        ok = (c[a, 0] != c[b, 0]) & (c[a, 1] != c[b, 1]) & (c[a, 2] != c[b, 2])
        for i, j in zip(a[ok], b[ok]):
            if len(ps) >= n:
                break
            ps.add(int(i), int(j), source_group="-".join(map(str, c[i])),
                   source_group_b="-".join(map(str, c[j])))
    return ps


# ---------------------------------------------------------------- similarity
TFIDF_SETTINGS = dict(
    lowercase=False,          # tokenize()가 이미 처리
    analyzer=lambda t: t,     # 미리 만든 token list를 그대로 사용
    min_df=2,                 # 1개 문서에만 나오는 token 제외
    max_df=0.6,               # 60% 이상 문서에 나오는 token 제외
    sublinear_tf=True,
    norm="l2",
)

def build_tfidf(body_tokens: list[list[str]]):
    """최종 train corpus 전체에 한 번만 fit. pair마다 다시 fit하지 않는다."""
    vec = TfidfVectorizer(**TFIDF_SETTINGS)
    X = vec.fit_transform(body_tokens)        # norm='l2' -> cosine = dot
    return vec, X


def similarity_frame(ps: PairSet, d: pd.DataFrame, title_sets, body_sets, X,
                     condition: str, method: str, start_id: int) -> pd.DataFrame:
    if not len(ps):
        return pd.DataFrame()
    f = pd.DataFrame(ps.rows)
    ia = f.idx_a.to_numpy(); ib = f.idx_b.to_numpy()
    f["title_jaccard"] = [jaccard(title_sets[a], title_sets[b]) for a, b in zip(ia, ib)]
    f["body_jaccard"]  = [jaccard(body_sets[a],  body_sets[b])  for a, b in zip(ia, ib)]
    f["body_tfidf_cosine"] = np.asarray(
        X[ia].multiply(X[ib]).sum(axis=1)).ravel()
    f.insert(0, "sample_id", [f"{method}_{condition}_{start_id + i:06d}"
                              for i in range(len(f))])
    f.insert(1, "condition", condition)
    f.insert(2, "sampling_method", method)
    for side, idx in (("a", ia), ("b", ib)):
        f[f"article_id_{side}"] = d.aid.to_numpy()[idx]
        for c in ("c1", "c2", "c3", "c4"):
            f[f"{c}_{side}"] = d[c].to_numpy()[idx]
    return f
