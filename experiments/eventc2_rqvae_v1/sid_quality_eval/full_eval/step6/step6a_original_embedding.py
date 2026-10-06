"""Step 6-A — original_embedding_cosine (checkpoint 불필요한 부분).

Step 3의 동일 pair set을 재사용한다 (재샘플링 없음).
reconstruction_cosine / self_reconstruction_cosine / delta는 checkpoint가 확보된 뒤
Step 6-B에서 같은 파일에 컬럼으로 추가한다.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
HERE = Path(__file__).resolve().parent
S3 = HERE.parent / "step3"
SP = Path("/tmp/claude-0/-home-user-SID-Project/203c7aef-bdd1-56e1-807f-a98088550c89/scratchpad/ec_eval/inputs")
SID = Path("/tmp/claude-0/-home-user-SID-Project/203c7aef-bdd1-56e1-807f-a98088550c89/scratchpad/ec_eval/inputs/sid_eventc2.parquet")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("입력 확인 ...")
    emb = np.load(SP / "article_embeddings.npy", mmap_mode="r")
    sid = pd.read_parquet(SID)
    sid["aid"] = sid.article_id.astype(str).str.strip()
    master = pd.concat([
        pd.read_parquet(SP / "master_modelinputs.parquet").assign(split="train"),
        pd.read_parquet(SP / "vmaster_modelinputs.parquet").assign(split="validation"),
    ], ignore_index=True)
    master["aid"] = master.article_id.astype(str).str.strip()
    print(f"  embeddings {emb.shape} {emb.dtype}  ·  SID {len(sid):,}  ·  master {len(master):,}")

    # ---- 매핑 검증 (실패 시 중단)
    j = sid.merge(master[["aid", "embedding_row", "model_category_id", "event_id"]],
                  on="aid", how="left", suffixes=("", "_m"), validate="1:1")
    checks = {
        "master에 전부 존재": int(j.embedding_row_m.notna().sum()),
        "embedding_row 일치": int((j.embedding_row == j.embedding_row_m).sum()),
        "c1 == model_category_id": int((j.c1 == j.model_category_id).sum()),
        "event_id 일치": int((j.event_id == j.event_id_m).sum()),
    }
    for k, v in checks.items():
        print(f"  {k}: {v:,}/{len(sid):,}")
    if any(v != len(sid) for v in checks.values()):
        raise SystemExit("매핑 검증 실패 — 중단합니다.")

    # ---- 기사별 원본 임베딩 추출 (12,860 × 768)
    order = sid.sort_values("article_id").reset_index(drop=True)
    rows = order.embedding_row.to_numpy(np.int64)
    X = np.asarray(emb[rows], dtype=np.float32)
    norm = np.linalg.norm(X, axis=1)
    print(f"\n  추출 {X.shape}  ·  L2 norm  min {norm.min():.4f} / median {np.median(norm):.4f} / max {norm.max():.4f}")
    print(f"  norm==0 인 기사: {int((norm == 0).sum())}")
    Xn = X / np.maximum(norm, 1e-12)[:, None]
    idx = {a: i for i, a in enumerate(order.article_id.astype(str))}

    # ---- Step 3의 동일 pair set에 컬럼 추가
    print("\nStep 3 pair set 로드 (재샘플링 없음) ...")
    pairs = pd.read_parquet(S3 / "quantitative_pairs.parquet")
    keep = ["pair_id", "analysis_set", "condition", "sampling_method",
            "article_id_a", "article_id_b", "c1_a", "c2_a", "c3_a", "c4_a",
            "c1_b", "c2_b", "c3_b", "c4_b",
            "title_jaccard", "body_jaccard", "body_tfidf_cosine"]
    out = pairs[keep].copy()
    ia = np.fromiter((idx[a] for a in out.article_id_a), np.int64, len(out))
    ib = np.fromiter((idx[b] for b in out.article_id_b), np.int64, len(out))
    out["original_embedding_cosine"] = np.einsum("ij,ij->i", Xn[ia], Xn[ib]).astype(np.float64)
    print(f"  {len(out):,} pair 계산 완료")

    # ---- 저장
    np.save(HERE / "original_embeddings.npy", X)
    order[["article_id", "split", "embedding_row", "c1", "c2", "c3", "c4"]].assign(
        row=np.arange(len(order)),
        original_l2_norm=norm,
    ).to_parquet(HERE / "reconstruction_index.parquet", index=False)
    out.to_parquet(HERE / "pair_embedding_cosine.parquet", index=False)

    # ---- summary
    def stats(v, n):
        v = v[~np.isnan(v)]
        return {"n_pairs": n, "count": int(v.size), "mean": float(v.mean()),
                "median": float(np.median(v)), "std": float(v.std(ddof=1)),
                "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
                "min": float(v.min()), "max": float(v.max())}
    rows_s = []
    for key, g in out.groupby(["analysis_set", "condition", "sampling_method"], sort=False):
        rows_s.append(dict(zip(["analysis_set", "condition", "sampling_method"], key),
                           metric="original_embedding_cosine",
                           **stats(g.original_embedding_cosine.to_numpy(), len(g))))
    summ = pd.DataFrame(rows_s)
    summ.round(6).to_csv(HERE / "embedding_cosine_summary.csv", index=False, encoding="utf-8-sig")
    summ.to_parquet(HERE / "embedding_cosine_summary.parquet", index=False)

    manifest = {
        "status": "partial — original_embedding_cosine만 계산. "
                  "reconstruction은 RQ-VAE checkpoint 확보 후 Step 6-B에서 추가.",
        "inputs": {
            "article_embeddings.npy": {
                "source": "git: origin/experiment/entity-linking-event "
                          ":data/output/exports/rqvae_train_inputs/article_embeddings.npy "
                          "(브랜치 내 6개 사본 모두 byte-identical, blob 0dde4c9f...)",
                "bytes": (SP / "article_embeddings.npy").stat().st_size,
                "sha256": sha256(SP / "article_embeddings.npy"),
                "shape": list(emb.shape), "dtype": str(emb.dtype),
            },
            "article_master.parquet": {
                "source": "git: origin/experiment/entity-linking-event:data/output/model_inputs/article_master.parquet",
                "bytes": (SP / "master_modelinputs.parquet").stat().st_size,
                "sha256": sha256(SP / "master_modelinputs.parquet"),
            },
            "validation_article_master.parquet": {
                "source": "git: origin/experiment/entity-linking-event:data/output/model_inputs/validation_article_master.parquet",
                "bytes": (SP / "vmaster_modelinputs.parquet").stat().st_size,
                "sha256": sha256(SP / "vmaster_modelinputs.parquet"),
            },
            "article_semantic_ids.parquet": {
                "source": str(SID), "bytes": SID.stat().st_size, "sha256": sha256(SID),
            },
        },
        "master_variant_selection": {
            "method": "SID 파일의 (embedding_row, c1, event_id, split)과 대조",
            "model_inputs": "12,860/12,860 전부 일치 → 채택",
            "exports/rqvae_*_inputs": "event_id 2,012/12,860만 일치 → 다른 event-linking 변종",
            "experiments/normalize_only": "event_id 2,490/12,860만 일치 → 변종",
            "note": "normalize_v2/model_inputs의 master는 model_inputs와 동일 blob이므로 "
                    "최종 파이프라인은 normalize_v2로 보인다.",
        },
        "still_missing_for_reconstruction": [
            "selected_checkpoint.json", "checkpoint_best_rec.pt (또는 선택된 variant)",
            "validation_event_c2_mapping.parquet (선택)",
        ],
        "pair_set": "Step 3 quantitative_pairs.parquet 그대로 재사용 (재샘플링 없음)",
        "n_pairs": int(len(out)), "n_articles": int(len(order)),
    }
    (HERE / "step6a_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 200)
    show = ["analysis_set", "condition", "sampling_method", "count", "mean",
            "median", "std", "q1", "q3", "min", "max"]
    print("\n=========== original_embedding_cosine ===========")
    print(summ[show].round(4).to_string(index=False))
    print(f"\n완료 → {HERE}")


if __name__ == "__main__":
    main()
