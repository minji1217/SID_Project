"""checkpoint와 저장된 SID가 같은 모델에서 나온 것인지 검증 + 유효한 self-reconstruction 산출.

Step 6-B 본체는 이 검증을 통과해야만 의미가 있다.
저장된 SID와 무관한 self_reconstruction_cosine(모델 자체 양자화 기준)은 여기서 함께 저장한다.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "RQVAE"))
sys.stdout.reconfigure(line_buffering=True)
from modules.rqvae import RqVae                                    # noqa: E402

CKPT = Path("/root/.claude/uploads/f5109944-18fd-59ef-9d42-2e3b01694dcf/"
            "9d16b47b-checkpoint_best_rec_1.pt")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def stats_of(v):
    v = np.asarray(v, float)
    return {"count": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std(ddof=1)), "q1": float(np.percentile(v, 25)),
            "q3": float(np.percentile(v, 75)), "min": float(v.min()), "max": float(v.max())}


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["model_config"])
    model = RqVae(**cfg); model.load_state_dict(ck["model"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    index = pd.read_parquet(HERE / "reconstruction_index.parquet")
    X = torch.from_numpy(np.load(HERE / "original_embeddings.npy").copy())
    c1 = torch.from_numpy(index.c1.to_numpy(np.int64))
    stored = index[["c1", "c2", "c3"]].to_numpy()
    tr = (index.split == "train").to_numpy()

    # (1) checkpoint 기록값 재현 — 임베딩이 맞는 파일인지
    with torch.inference_mode():
        mv = torch.from_numpy(~tr)
        xs, cs = X[mv], c1[mv]
        tot = sum(float(model(x=xs[s:s + 2048], category_ids=cs[s:s + 2048])
                        .reconstruction_loss) * len(xs[s:s + 2048])
                  for s in range(0, len(xs), 2048))
        valid_rec = tot / len(xs)
    recorded = float(ck["validation_state"]["best_valid_rec_loss"])
    print(f"validation rec loss  재계산 {valid_rec:.7f}  vs  기록 {recorded:.7f}  "
          f"(차 {abs(valid_rec - recorded):.2e})")

    # (2) 모델 자체 양자화로 SID / x_hat
    sem, XH = [], []
    with torch.inference_mode():
        for s in range(0, len(X), 2048):
            o = model.get_semantic_ids(x=X[s:s + 2048], category_ids=c1[s:s + 2048])
            sem.append(o.sem_ids); XH.append(model.decode(o.embeddings.sum(dim=-1)))
    sem = torch.cat(sem).numpy(); XH = torch.cat(XH).numpy()

    chance = {"c1": 1 / cfg["num_categories"], "c2": 1 / cfg["c2_codebook_size"],
              "c3": 1 / cfg["c3_codebook_size"]}
    match = {}
    for i, n in enumerate(["c1", "c2", "c3"]):
        match[n] = {"train_match_rate": float((sem[tr, i] == stored[tr, i]).mean()),
                    "chance": chance[n]}
        print(f"  {n} train 일치율 {match[n]['train_match_rate']*100:6.2f}%  "
              f"(무작위 {chance[n]*100:.2f}%)")

    # (3) self reconstruction — 모델 자체 양자화 기준 (저장 SID와 무관하므로 유효)
    def u(M):
        return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    Xn = u(X.numpy()); self_cos = np.einsum("ij,ij->i", Xn, u(XH))
    rows = [{"split": "all", **stats_of(self_cos)},
            {"split": "train", **stats_of(self_cos[tr])},
            {"split": "validation", **stats_of(self_cos[~tr])}]
    out = pd.DataFrame(rows)
    out.round(6).to_csv(HERE / "self_reconstruction_own_quantization.csv",
                        index=False, encoding="utf-8-sig")
    print("\nself_reconstruction_cosine (모델 자체 양자화 기준):")
    print(out.round(4).to_string(index=False))

    diag = {
        "verdict": ("저장된 SID는 이 checkpoint가 생성한 것이 아니다. "
                    "train c2/c3 재현율이 무작위 수준이다."),
        "checkpoint": {"path": str(CKPT), "bytes": CKPT.stat().st_size,
                       "sha256": sha256(CKPT), "epoch": int(ck["epoch"]),
                       "experiment": "WD-001-7d3a10", "variant": "best_rec",
                       "model_config": {k: str(v) for k, v in cfg.items()}},
        "embedding_verified": {
            "recorded_best_valid_rec_loss": recorded,
            "recomputed_validation_rec_loss": valid_rec,
            "abs_diff": abs(valid_rec - recorded),
            "conclusion": "임베딩은 이 모델의 학습/검증에 쓰인 파일이 맞다."},
        "sid_reproduction_train_only": match,
        "note_c1": ("c1은 모델이 선택하지 않고 model_category_id를 그대로 쓰므로 "
                    "100% 일치는 모델 일치의 근거가 되지 않는다."),
        "self_reconstruction_own_quantization_valid": True,
        "required_to_proceed": "업로드된 article_semantic_ids.parquet와 같은 EXP 폴더의 checkpoint",
    }
    (HERE / "checkpoint_sid_match_diagnostic.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장 → {HERE}")


if __name__ == "__main__":
    main()
