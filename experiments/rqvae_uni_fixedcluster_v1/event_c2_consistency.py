"""Train 사건(기사 2개 이상) 안에서 기사별 c2 가 얼마나 같은지 잰다. 학습 없이 checkpoint 로 평가만 한다.

c2 는 학습 때와 같은 방식(기사별 r1 = h - Q1[c1] 최근접 Q2 코드)으로 뽑는다.
사건은 전처리 Train 사건(article_master.event_id)을 쓴다.

지표 (모델마다)
  all_same      사건 안 기사의 c2 가 전부 같은 사건 비율
  mode_share    사건마다 가장 많은 c2 를 가진 기사 수의 합 ÷ 기사 수 (사건 안 c2 다수결 일치율)
  pair_same     사건 안 기사 쌍 중 c2 가 같은 쌍 비율
  pair_same_c12 사건 안 기사 쌍 중 (c1, c2) 가 같은 쌍 비율
  baseline      우연 수준: 서로 다른 사건이지만 c1(카테고리)이 같은 Train 기사 쌍에서 c2 가 같은 비율 (무작위 20만 쌍)
                72h 안 + 같은 c1 인 다른 사건 쌍 기준도 같이 계산 (시기가 비슷한 기사끼리의 우연 수준)
  크기별(2 / 3~5 / 6+) pair_same

사용 (AWS):
  python experiments/rqvae_uni_fixedcluster_v1/event_c2_consistency.py \\
      --code_dir ~/minji/rqvae_uni_code_127ae7e/RQVAE \\
      --input_dir data/output/fixedcluster_v1/exports/rqvae_inputs \\
      --ckpt ver1=~/minji/checkpoint_best_rec.pt \\
      --ckpt ver2_s42=data/output/rqvae/uni_fixedcluster_lu0.05_m0.5_s42/checkpoint_best_rec.pt \\
      --out data/output/rqvae/event_c2_consistency
"""
import argparse
import inspect
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="이름=경로 (여러 번)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.code_dir).expanduser().resolve()))
    import torch
    from modules.rqvae import RqVae

    d = Path(a.input_dir).expanduser()
    tr = pd.read_parquet(d / "article_master.parquet").sort_values("article_id").reset_index(drop=True)
    emb = np.load(d / "article_embeddings.npy", mmap_mode="r")
    size = tr.groupby("event_id").article_id.transform("size")
    multi = tr[size >= 2]
    print(f"[사건] Train 사건 {tr.event_id.nunique():,} · 기사 2개 이상 사건 {multi.event_id.nunique():,} (기사 {len(multi):,})")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)

    # 우연 수준 비교용 쌍: 다른 사건 · 같은 c1 (무작위) / 다른 사건 · 같은 c1 · 발행 72h 안
    c1 = tr.model_category_id.to_numpy()
    ev = tr.event_id.to_numpy()
    t = pd.to_datetime(tr.published_time).to_numpy().astype("datetime64[s]").astype(np.int64)
    n = len(tr)
    i = rng.integers(0, n, 600_000)
    j = rng.integers(0, n, 600_000)
    ok = (i != j) & (ev[i] != ev[j]) & (c1[i] == c1[j])
    rand_pairs = (i[ok][:200_000], j[ok][:200_000])
    order = np.argsort(t)
    ts = t[order]
    near_i, near_j = [], []
    for k in rng.integers(0, n, 400_000):
        p = np.searchsorted(ts, t[k] - 72 * 3600)
        q = np.searchsorted(ts, t[k] + 72 * 3600, side="right")
        if q - p < 2:
            continue
        m = order[rng.integers(p, q)]
        if m != k and ev[m] != ev[k] and c1[m] == c1[k]:
            near_i.append(k)
            near_j.append(m)
        if len(near_i) >= 100_000:
            break
    near_pairs = (np.array(near_i), np.array(near_j))

    groups = [g.index.to_numpy() for _, g in multi.groupby("event_id")]
    pairs = [(x, y) for g in groups for x, y in combinations(g, 2)]
    pi = np.array([p[0] for p in pairs])
    pj = np.array([p[1] for p in pairs])
    gsize = {}
    for g in groups:
        for x in g:
            gsize[x] = len(g)
    bucket = lambda s: "2" if s == 2 else ("3~5" if s <= 5 else "6+")

    report = {"n_events_total": int(tr.event_id.nunique()), "n_multi_events": len(groups), "n_multi_articles": int(len(multi)),
              "n_pairs": len(pairs), "models": {}}
    rows = []
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        ck = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
        acc = set(inspect.signature(RqVae.__init__).parameters) - {"self"}
        model = RqVae(**{k: v for k, v in dict(ck["model_config"]).items() if k in acc})
        model.load_state_dict(ck["model"], strict=True)
        model.to(dev).eval()
        c2 = np.empty(n, dtype=np.int64)
        with torch.no_grad():
            for s in range(0, n, a.batch_size):
                part = tr.iloc[s:s + a.batch_size]
                x = torch.from_numpy(np.asarray(emb[part.embedding_row.to_numpy()], dtype=np.float32)).to(dev)
                c = torch.from_numpy(part.model_category_id.to_numpy().astype(np.int64)).to(dev)
                c2[s:s + len(part)] = model.get_semantic_ids(x=x, category_ids=c).sem_ids[:, 1].cpu().numpy()

        all_same = np.mean([len(set(c2[g])) == 1 for g in groups])
        mode_share = sum(np.bincount(c2[g]).max() for g in groups) / len(multi)
        pair_same = float(np.mean(c2[pi] == c2[pj]))
        pair_same_c12 = float(np.mean((c2[pi] == c2[pj]) & (c1[pi] == c1[pj])))
        base_rand = float(np.mean(c2[rand_pairs[0]] == c2[rand_pairs[1]]))
        base_near = float(np.mean(c2[near_pairs[0]] == c2[near_pairs[1]]))
        by = {}
        for b in ["2", "3~5", "6+"]:
            sel = np.array([bucket(gsize[x]) == b for x in pi])
            by[b] = {"pairs": int(sel.sum()), "pair_same": float(np.mean(c2[pi][sel] == c2[pj][sel])) if sel.any() else None}
        r = {"epoch": int(ck.get("epoch", -1)) + 1, "all_same": float(all_same), "mode_share": float(mode_share),
             "pair_same": pair_same, "pair_same_c12": pair_same_c12,
             "baseline_same_c1_random": base_rand, "baseline_same_c1_within72h": base_near, "pair_same_by_size": by}
        report["models"][name] = r
        rows.append({"model": name, **{k: v for k, v in r.items() if k != "pair_same_by_size"},
                     **{f"pair_same_size{b}": v["pair_same"] for b, v in by.items()}})
        print(f"[{name}] epoch {r['epoch']} · 전부 같은 사건 {all_same:.1%} · 다수결 일치 {mode_share:.1%} · "
              f"쌍 일치 {pair_same:.1%} (우연: 같은 c1 {base_rand:.1%} · 72h 안 같은 c1 {base_near:.1%}) · "
              f"크기별 " + " / ".join(f"{b}: {v['pair_same']:.1%}" for b, v in by.items() if v['pair_same'] is not None))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "event_c2_consistency.csv", index=False)
    (out / "event_c2_consistency.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[쌍] 사건 안 쌍 {len(pairs):,} · 비교 쌍: 같은 c1 무작위 {len(rand_pairs[0]):,} · 72h 안 같은 c1 {len(near_pairs[0]):,}")
    print(f"저장: {out}/event_c2_consistency.csv, .json")


if __name__ == "__main__":
    main()
