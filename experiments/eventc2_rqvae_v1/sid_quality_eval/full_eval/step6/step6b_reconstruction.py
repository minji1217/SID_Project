"""Step 6-B — RQ-VAE reconstruction 기반 정량평가 (교수님 피드백 #3).

checkpoint: selected_checkpoint.json 기준 best_rec (experiment WD-001-7d3a10).
model_config는 checkpoint 내부 값을 권위로 사용한다 (base gin의 c3=256은 쓰지 않는다).

Step 3의 pair set 207,670개를 그대로 재사용한다 (재샘플링 없음).
Step 6-A에서 계산한 original_embedding_cosine은 그대로 유지하고 컬럼만 추가한다.

계산
  1 original_embedding_cosine     (Step 6-A 유지)
  2 reconstruction_cosine         cos(x_hat_i, x_hat_j)
  3 self_reconstruction_cosine    cos(x_i, x_hat_i)  — article별, train/validation 분리
  4 similarity preservation       delta / abs_delta + Pearson / Spearman
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = Path("/home/user/SID_Project")
sys.path.insert(0, str(REPO / "RQVAE"))
sys.path.insert(0, str(ROOT / "step3"))
sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from modules.rqvae import RqVae                                    # noqa: E402
from step3_quantitative import (                                   # noqa: E402
    CONDITIONS, COND_COLOR, SURFACE, INK, INK2, GRID,
)

CKPT = Path("/tmp/claude-0/-home-user-SID-Project/203c7aef-bdd1-56e1-807f-a98088550c89/scratchpad/ec_eval/inputs/eventc2_best_rec.pt")
EXPERIMENT = "eventc2_lu0.05_m0.5"   # UNI(lu0.05, m0.5) + c2_mode=event
SP = Path("/tmp/claude-0/-home-user-SID-Project/203c7aef-bdd1-56e1-807f-a98088550c89/scratchpad/ec_eval/inputs")
COND_LABEL = {"prefix_3_same": "prefix_3\n(c1,c2,c3 동일)",
              "prefix_2_same": "prefix_2\n(c1,c2 동일)",
              "semantic_different": "semantic_different\n(c1,c2,c3 모두 다름)"}
COND_SHORT = {"prefix_3_same": "prefix_3", "prefix_2_same": "prefix_2",
              "semantic_different": "semantic_different"}

plt.rcParams.update({
    "font.family": "WenQuanYi Zen Hei",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
})


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def stats_of(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {k: float("nan") for k in
                ("count", "mean", "median", "std", "q1", "q3", "min", "max")}
    return {"count": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
            "min": float(v.min()), "max": float(v.max())}


def unit(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-12)


def main():
    guard = {p.name for p in HERE.iterdir()} & {"pair_similarity_preservation.parquet"}
    if guard:
        raise SystemExit(f"Step 6-B 결과가 이미 있습니다: {sorted(guard)}")

    # ---------------------------------------------------------------- 모델
    print("checkpoint 로드 ...")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["model_config"])
    # uniqueness loss 파라미터(lambda_uniq, uniqueness_margin)는 학습 전용이며
    # encoder/decoder/quantizer 구조를 바꾸지 않는다. base RqVae 생성 시 제외한다.
    import inspect as _inspect
    accepted = set(_inspect.signature(RqVae.__init__).parameters) - {"self"}
    train_only = {k: v for k, v in cfg.items() if k not in accepted}
    model = RqVae(**{k: v for k, v in cfg.items() if k in accepted})
    missing, unexpected = model.load_state_dict(ck["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    if train_only:
        print(f"  (학습 전용 파라미터 제외: {train_only})")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  epoch {ck['epoch']}  global_step {ck['global_step']}")
    print(f"  input_dim {cfg['input_dim']} · embed_dim {cfg['embed_dim']} · "
          f"hidden {cfg['hidden_dims']} · c1 {cfg['num_categories']} · "
          f"c2 {cfg['c2_codebook_size']} · c3 {cfg['c3_codebook_size']}")

    # ---------------------------------------------------------------- 데이터
    print("\n입력 로드 ...")
    index = pd.read_parquet(HERE / "reconstruction_index.parquet")
    X = np.load(HERE / "original_embeddings.npy")                  # [12860, 768]
    print(f"  article {len(index):,}  ·  original {X.shape}")
    assert (index.row.to_numpy() == np.arange(len(index))).all()

    c1 = torch.from_numpy(index.c1.to_numpy(np.int64))
    c2 = torch.from_numpy(index.c2.to_numpy(np.int64))
    c3 = torch.from_numpy(index.c3.to_numpy(np.int64))

    # ---------------------------------------------------------------- x_hat
    print("\nx_hat 생성 — 저장된 (c1,c2,c3)에서 직접 ...")
    with torch.inference_mode():
        q1 = model.quantizer_1.get_item_embeddings(c1)
        q2 = model.quantizer_2.get_item_embeddings(c2)
        q3 = model.quantizer_3.get_item_embeddings(c3)
        z_q = q1 + q2 + q3
        Xh = model.decode(z_q).numpy().astype(np.float32)
    print(f"  z_q {tuple(z_q.shape)}  ->  x_hat {Xh.shape}")

    # ---------------------------------------------------------------- 검증
    print("\n=== 검증 ===")
    verify = {}

    # (1) train SID 재현 — encoder/quantizer 정식 경로
    with torch.inference_mode():
        Xt = torch.from_numpy(X)
        # EventC2: event_ids를 넘겨 event 규칙으로 재현한다 (기사 전체를 한 번에 넣어 event가 쪼개지지 않게)
        sid_ev = pd.read_parquet(SP / "sid_eventc2.parquet")[["article_id", "event_id"]]
        sid_ev["article_id"] = sid_ev.article_id.astype(str)
        ev = index[["article_id"]].astype(str).merge(sid_ev, on="article_id", how="left", validate="1:1")
        event_ids = torch.from_numpy(ev.event_id.to_numpy(np.int64))
        table = {int(k): int(v) for k, v in ck["event_code_table"].items()}
        is_val = torch.from_numpy((index.split == "validation").to_numpy())
        sem = np.zeros((len(Xt), 3), dtype=np.int64)
        for mask, override in ((~is_val, None), (is_val, True)):
            ov = None
            if override:
                ov = torch.tensor([table.get(int(e), -1) for e in event_ids[mask].tolist()])
            o = model.get_semantic_ids(x=Xt[mask], category_ids=c1[mask],
                                       event_ids=event_ids[mask], event_code_override=ov)
            sem[mask.numpy()] = o.sem_ids.numpy()
    stored = index[["c1", "c2", "c3"]].to_numpy()
    for sp in ("train", "validation"):
        m = (index.split == sp).to_numpy()
        eq = (sem[m] == stored[m])
        verify[f"sid_reproduce_{sp}"] = {
            "n": int(m.sum()),
            "c1_match": int(eq[:, 0].sum()), "c2_match": int(eq[:, 1].sum()),
            "c3_match": int(eq[:, 2].sum()),
            "all3_match": int(eq.all(axis=1).sum()),
        }
        v = verify[f"sid_reproduce_{sp}"]
        print(f"  SID 재현 [{sp}] n={v['n']:,}  c1 {v['c1_match']:,} · "
              f"c2 {v['c2_match']:,} · c3 {v['c3_match']:,} · 전부일치 {v['all3_match']:,}")
    print("    (EventC2: train은 현재 encoder의 EventCode, validation은 checkpoint EventCode 표 상속으로 재현)")

    # (2) 같은 (c1,c2,c3) -> 동일 x_hat
    key = pd.Series(list(map(tuple, stored)))
    dup = key.duplicated(keep=False)
    maxdiff = 0.0
    for _, g in index[dup.to_numpy()].groupby(["c1", "c2", "c3"]):
        r = g.row.to_numpy()
        if len(r) > 1:
            maxdiff = max(maxdiff, float(np.abs(Xh[r] - Xh[r[0]]).max()))
    verify["identical_xhat_for_same_sid"] = {
        "groups_checked": int(index[dup.to_numpy()].groupby(["c1", "c2", "c3"]).ngroups),
        "max_abs_diff": maxdiff}
    print(f"  동일 SID 기사들의 x_hat 최대 절대차: {maxdiff:.3e}  "
          f"({verify['identical_xhat_for_same_sid']['groups_checked']} group)")

    # (3) forward()의 reconstruction_loss와 교차 검증
    with torch.inference_mode():
        tr = ~is_val
        fo = model(x=Xt[tr], category_ids=c1[tr], event_ids=event_ids[tr])
        rec_forward = float(fo.reconstruction_loss)
        own = model.reconstruction_loss_fn(x_hat=torch.from_numpy(Xh[tr.numpy()]),
                                           x=Xt[tr]).mean().item()
    verify["forward_rec_loss_crosscheck"] = {
        "forward": rec_forward, "from_stored_sid_xhat": own,
        "abs_diff": abs(rec_forward - own)}
    print(f"  forward() rec loss {rec_forward:.6f}  vs  저장 SID x_hat 기준 {own:.6f}  "
          f"(차 {abs(rec_forward - own):.2e})")

    # ---------------------------------------------------------------- 3. self recon
    print("\nself_reconstruction_cosine ...")
    Xn, Xhn = unit(X), unit(Xh)
    self_cos = np.einsum("ij,ij->i", Xn, Xhn)
    index = index.copy()
    index["self_reconstruction_cosine"] = self_cos
    index["xhat_l2_norm"] = np.linalg.norm(Xh, axis=1)
    self_rows = [{"split": "all", **stats_of(self_cos)}]
    for sp in ("train", "validation"):
        self_rows.append({"split": sp,
                          **stats_of(self_cos[(index.split == sp).to_numpy()])})
    self_summary = pd.DataFrame(self_rows)
    print(self_summary.round(4).to_string(index=False))

    # ---------------------------------------------------------------- 2·4 pair
    print("\npair 지표 — Step 3 pair set 재사용 ...")
    pairs = pd.read_parquet(HERE / "pair_embedding_cosine.parquet")
    aid2row = dict(zip(index.article_id.astype(str), index.row.to_numpy()))
    ia = np.fromiter((aid2row[a] for a in pairs.article_id_a), np.int64, len(pairs))
    ib = np.fromiter((aid2row[b] for b in pairs.article_id_b), np.int64, len(pairs))
    pairs["reconstruction_cosine"] = np.einsum("ij,ij->i", Xhn[ia], Xhn[ib])
    pairs["delta_cosine"] = pairs.reconstruction_cosine - pairs.original_embedding_cosine
    pairs["abs_delta_cosine"] = pairs.delta_cosine.abs()
    print(f"  {len(pairs):,} pair 계산 완료")

    MAIN = pairs[(pairs.sampling_method.isin(["exhaustive", "group_balanced"]))
                 & (pairs.analysis_set.isin(["large", "large_and_balanced"]))]
    p3 = MAIN[MAIN.condition == "prefix_3_same"]
    verify["prefix3_reconstruction_cosine"] = {
        "n": int(len(p3)), "min": float(p3.reconstruction_cosine.min()),
        "max": float(p3.reconstruction_cosine.max()),
        "max_abs_dev_from_1": float((1.0 - p3.reconstruction_cosine).abs().max())}
    print(f"  prefix_3_same reconstruction_cosine: "
          f"[{p3.reconstruction_cosine.min():.12f}, {p3.reconstruction_cosine.max():.12f}] "
          f"— 1.0에서의 최대 편차 {(1.0 - p3.reconstruction_cosine).abs().max():.3e}")

    # summary
    srows = []
    for key_, g in pairs.groupby(["analysis_set", "condition", "sampling_method"], sort=False):
        for met in ("original_embedding_cosine", "reconstruction_cosine",
                    "delta_cosine", "abs_delta_cosine"):
            srows.append(dict(zip(["analysis_set", "condition", "sampling_method"], key_),
                              metric=met, n_pairs=len(g), **stats_of(g[met].to_numpy())))
    summary = pd.DataFrame(srows)

    # correlation
    print("\ncorrelation (original vs reconstruction) ...")
    crows = []
    for key_, g in pairs.groupby(["analysis_set", "condition", "sampling_method"], sort=False):
        o = g.original_embedding_cosine.to_numpy()
        r = g.reconstruction_cosine.to_numpy()
        if np.std(r) < 1e-12 or np.std(o) < 1e-12:
            pr = sr = float("nan"); note = "reconstruction_cosine이 상수(≈1.0)라 정의되지 않음"
            pp = sp_ = float("nan")
        else:
            pr, pp = stats.pearsonr(o, r)
            sr, sp_ = stats.spearmanr(o, r)
            note = ""
        crows.append(dict(zip(["analysis_set", "condition", "sampling_method"], key_),
                          n_pairs=len(g), pearson_r=pr, pearson_p=pp,
                          spearman_rho=sr, spearman_p=sp_, note=note))
    corr = pd.DataFrame(crows)
    show_c = ["condition", "sampling_method", "n_pairs", "pearson_r", "spearman_rho", "note"]
    print(corr[corr.analysis_set.isin(["large", "large_and_balanced"])][show_c]
          .round(4).to_string(index=False))

    # ---------------------------------------------------------------- 저장
    print("\n저장 ...")
    np.save(HERE / "reconstructed_embeddings.npy", Xh)
    index.to_parquet(HERE / "reconstruction_index.parquet", index=False)
    pairs.to_parquet(HERE / "pair_similarity_preservation.parquet", index=False)
    summary.round(8).to_csv(HERE / "embedding_cosine_summary.csv", index=False,
                            encoding="utf-8-sig")
    summary.to_parquet(HERE / "embedding_cosine_summary.parquet", index=False)
    self_summary.round(8).to_csv(HERE / "self_reconstruction_summary.csv", index=False,
                                 encoding="utf-8-sig")
    corr.round(8).to_csv(HERE / "similarity_preservation_correlation.csv", index=False,
                         encoding="utf-8-sig")

    manifest = {
        "checkpoint": {
            "selected_variant": "best_rec",
            "experiment": EXPERIMENT,
            "training_only_params_excluded_from_model": {k: str(v) for k, v in
                                                         train_only.items()},
            "uploaded_path": str(CKPT),
            "bytes": CKPT.stat().st_size,
            "sha256": sha256(CKPT),
            "epoch": int(ck["epoch"]), "global_step": int(ck["global_step"]),
            "model_config": {k: str(v) for k, v in cfg.items()},
            "validation_state": {k: float(v) for k, v in ck["validation_state"].items()},
            "state_dict_shapes": {k: list(v.shape) for k, v in ck["model"].items()},
            "note": "base gin의 c3=256은 사용하지 않았다. checkpoint의 model_config가 권위.",
        },
        "embeddings": {
            "article_embeddings.npy sha256":
                sha256(SP / "article_embeddings.npy"),
            "original_l2_normalized": True,
        },
        "forward_path": "x_hat = decoder(Q1[c1] + Q2[c2] + Q3[c3]) — 저장된 SID 정수 사용",
        "pair_set": "Step 3 quantitative_pairs.parquet 그대로 (재샘플링 없음)",
        "n_articles": int(len(index)), "n_pairs": int(len(pairs)),
        "verification": verify,
        "structural_note": ("prefix_3_same은 (c1,c2,c3)가 같으므로 q1+q2+q3가 동일하고 "
                            "decoder 입력이 동일하다. 따라서 reconstruction_cosine = 1.0은 "
                            "모델 구조상 자명한 값이며 학습 성공의 근거가 아니다."),
    }
    (HERE / "step6b_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------------------- 그림
    print("\n그림 ...")
    # fig9: condition별 original vs reconstruction
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2))
    for ax, met, title in zip(
            axes, ["original_embedding_cosine", "reconstruction_cosine", "abs_delta_cosine"],
            ["original embedding cosine", "reconstruction cosine",
             "|reconstruction − original|"]):
        vals, labs, cols = [], [], []
        for c in CONDITIONS:
            v = MAIN[MAIN.condition == c][met].dropna().to_numpy()
            if v.size:
                vals.append(v); labs.append(f"{COND_LABEL[c]}\nn={v.size:,}"); cols.append(COND_COLOR[c])
        pos = np.arange(1, len(vals) + 1)
        bp = ax.boxplot(vals, positions=pos, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.6),
                        whiskerprops=dict(color=INK2, linewidth=1.0),
                        capprops=dict(color=INK2, linewidth=1.0))
        for b, c in zip(bp["boxes"], cols):
            b.set(facecolor=c, edgecolor=SURFACE, linewidth=2.0, alpha=0.92)
        lo = min(np.percentile(v, 1) for v in vals); hi = max(np.percentile(v, 99) for v in vals)
        pad = max((hi - lo) * 0.30, 1e-4)
        ax.set_ylim(lo - pad * 0.25, hi + pad)
        for p_, v in zip(pos, vals):
            ax.annotate(f"med {np.median(v):.4f}\nmean {v.mean():.4f}",
                        xy=(p_, hi + pad * 0.92), ha="center", va="top",
                        fontsize=7.6, color=INK2, linespacing=1.35)
        ax.set_xticks(pos); ax.set_xticklabels(labs, fontsize=7.6)
        ax.set_title(title, pad=8); ax.set_ylabel(title)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    fig.suptitle("fig9 — 임베딩 공간 유사도 (Large-sample · group-balanced)", fontsize=12.5, y=0.985)
    fig.text(0.5, 0.012, "prefix_3_same의 reconstruction cosine = 1.0은 동일 code triple에서 "
                         "오는 구조적 결과이며 성능 근거가 아니다.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(HERE / "fig9_embedding_cosine.png", dpi=170); plt.close(fig)
    print("  fig9_embedding_cosine.png")

    # fig10: original vs reconstruction 산점 (hexbin) — prefix_2 / semantic_different
    cs = ["prefix_2_same", "semantic_different"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))
    for ax, c in zip(axes, cs):
        g = MAIN[MAIN.condition == c]
        hb = ax.hexbin(g.original_embedding_cosine, g.reconstruction_cosine,
                       gridsize=55, bins="log", mincnt=1, cmap="Blues", linewidths=0)
        lo = min(g.original_embedding_cosine.min(), g.reconstruction_cosine.min())
        hi = max(g.original_embedding_cosine.max(), g.reconstruction_cosine.max())
        ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1.2, linestyle="--", label="y = x")
        r = corr[(corr.condition == c) & (corr.analysis_set == "large")
                 & (corr.sampling_method == "group_balanced")].iloc[0]
        ax.set_title(f"{COND_SHORT[c]}  ·  Pearson {r.pearson_r:.3f} / Spearman {r.spearman_rho:.3f}",
                     fontsize=10, pad=7)
        ax.set_xlabel("original embedding cosine"); ax.set_ylabel("reconstruction cosine")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        fig.colorbar(hb, ax=ax, label="pair 수 (log)")
        ax.grid(True, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("fig10 — 원본 공간의 기사 간 관계가 reconstruction에 보존되는가",
                 fontsize=12.5, y=0.985)
    fig.text(0.5, 0.012, "prefix_3_same은 reconstruction cosine이 상수 1.0이라 상관계수가 "
                         "정의되지 않으므로 제외.", ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(HERE / "fig10_similarity_preservation.png", dpi=170); plt.close(fig)
    print("  fig10_similarity_preservation.png")

    # fig11: self reconstruction cosine
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    for sp, col in (("train", COND_COLOR["prefix_3_same"]),
                    ("validation", COND_COLOR["prefix_2_same"])):
        v = self_cos[(index.split == sp).to_numpy()]
        axes[0].hist(v, bins=70, alpha=0.72, color=col, label=f"{sp} (n={v.size:,})",
                     edgecolor=SURFACE, linewidth=0.4)
    axes[0].set_xlabel("self_reconstruction_cosine  cos(x, x̂)")
    axes[0].set_ylabel("기사 수"); axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("분포", pad=8); axes[0].grid(True, axis="y", alpha=0.5)
    axes[0].set_axisbelow(True)
    vals = [self_cos[(index.split == sp).to_numpy()] for sp in ("train", "validation")]
    bp = axes[1].boxplot(vals, positions=[1, 2], widths=0.5, patch_artist=True,
                         showfliers=False, medianprops=dict(color=INK, linewidth=1.6),
                         whiskerprops=dict(color=INK2, linewidth=1.0),
                         capprops=dict(color=INK2, linewidth=1.0))
    for b, c in zip(bp["boxes"], [COND_COLOR["prefix_3_same"], COND_COLOR["prefix_2_same"]]):
        b.set(facecolor=c, edgecolor=SURFACE, linewidth=2.0, alpha=0.92)
    hi = max(np.percentile(v, 99) for v in vals); lo = min(np.percentile(v, 1) for v in vals)
    axes[1].set_ylim(lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.42)
    for p_, v in zip([1, 2], vals):
        axes[1].annotate(f"med {np.median(v):.4f}\nmean {v.mean():.4f}",
                         xy=(p_, hi + (hi - lo) * 0.38), ha="center", va="top",
                         fontsize=8, color=INK2, linespacing=1.35)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels([f"train\nn={vals[0].size:,}", f"validation\nn={vals[1].size:,}"],
                            fontsize=8.5)
    axes[1].set_ylabel("self_reconstruction_cosine"); axes[1].set_title("split 비교", pad=8)
    axes[1].yaxis.grid(True); axes[1].set_axisbelow(True)
    fig.suptitle("fig11 — RQ-VAE가 개별 기사 임베딩을 얼마나 복원하는가  cos(x, x̂)",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(HERE / "fig11_self_reconstruction.png", dpi=170); plt.close(fig)
    print("  fig11_self_reconstruction.png")

    pd.set_option("display.width", 220)
    sh = ["condition", "sampling_method", "metric", "count", "mean", "median", "std", "q1", "q3"]
    HEADS = summary[(summary.sampling_method.isin(["exhaustive", "group_balanced"]))
                    & (summary.analysis_set.isin(["large", "large_and_balanced"]))]
    print("\n================ pair 지표 (Large · group-balanced) ================")
    print(HEADS.sort_values(["metric", "condition"])[sh].round(5).to_string(index=False))
    print(f"\n완료 → {HERE}")


if __name__ == "__main__":
    main()
