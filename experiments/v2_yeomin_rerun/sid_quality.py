"""RQ-VAE · SID 품질 표: 재구성 손실(train / validation)과 SID 충돌, 코드 사용 수를 split별로.

    python experiments/v2_yeomin_rerun/sid_quality.py                # AWS 기본 경로
    python experiments/v2_yeomin_rerun/sid_quality.py --no-rec       # 재구성 손실 빼고 (GPU 없이 몇 초)

정의 (여민 RQVAE/evaluate 와 같음)
  재구성 손실  model(x, category).reconstruction_loss 의 기사 평균 (gumbel_t 0.001, 여민 eval_reconstruction.py 함수 그대로)
               train = article_master (RQ-VAE 학습 기사), validation = validation_article_master (best checkpoint 고르는 기사)
               c2 는 기사마다 가장 가까운 코드라서, SID 파일의 "사건 C2"와는 다를 수 있다 (모델 자체의 복원 능력)
  충돌률       (c1, c2, c3) 가 같은 기사가 둘 이상인 기사의 비율 (keep=False, 여민 eval_sid_collision.py)
  c4 > 0       충돌 묶음에서 두 번째부터의 기사 비율 (첫 기사는 c4 = 0)
  코드 사용 수 그 split 기사들이 실제로 쓴 c1 / c2 / c3 종류, (c1, c2) 조합 수, (c1, c2, c3) 조합 수

입력 (읽기만)
  SID   <V2>/sid/{h,r}/article_semantic_ids.parquet, V1 = ~/minji/uni_post_rqvae/article_semantic_ids.parquet
  모델  <V2>/rqvae/*/checkpoint_best_rec.pt (있는 것 전부)
  데이터 <PRE>/exports/rqvae_inputs/ (article_master, validation_article_master, article_embeddings.npy)
결과: 화면 + <V2>/analysis/sid_quality.json (이미 있으면 _2, _3 … 로)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CODE = {"c1": 25, "c2": 128, "c3": 512}


def sid_stats(a: pd.DataFrame) -> dict:
    n = len(a)
    coll = a.duplicated(["c1", "c2", "c3"], keep=False)
    return {
        "기사": n,
        "c1 종류": int(a.c1.nunique()), "c2 종류": int(a.c2.nunique()), "c3 종류": int(a.c3.nunique()),
        "(c1,c2) 조합": int(a[["c1", "c2"]].drop_duplicates().shape[0]),
        "(c1,c2,c3) 조합": int(a[["c1", "c2", "c3"]].drop_duplicates().shape[0]),
        "충돌률": float(coll.mean()) if n else float("nan"),
        "c4>0 비율": float((a.c4 > 0).mean()) if n else float("nan"),
        "c4 최댓값": int(a.c4.max()) if n else 0,
    }


def load_sid(path: Path, inp: Path) -> pd.DataFrame:
    a = pd.read_parquet(path)
    if "split" not in a.columns:   # 예전 SID 파일: 지금 전처리 기사 목록으로는 나눌 수 없음
        a["split"] = "all"
    return a


def rec_losses(ym: Path, inp: Path, ckpts: list, batch: int) -> dict:
    sys.path.insert(0, str(ym / "RQVAE"))
    sys.path.insert(0, str(ym / "RQVAE" / "evaluate"))
    import torch
    from eval_reconstruction import compute_reconstruction_loss, load_model   # 여민 코드 그대로
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = inp / "article_embeddings.npy"
    out = {}
    for ck in ckpts:
        name = ck.parent.name
        try:
            m = load_model(ck, dev)
            tr = compute_reconstruction_loss(m, inp / "article_master.parquet", emb, dev, batch_size=batch)
            va = compute_reconstruction_loss(m, inp / "validation_article_master.parquet", emb, dev, batch_size=batch)
            out[name] = {"train": tr, "validation": va}
            summ = ck.parent / "summary.json"
            if summ.exists():
                s = json.loads(summ.read_text())
                out[name]["best_rec (학습 기록)"] = s.get("best_rec", s.get("best_valid_rec_loss"))
                out[name]["best_rec_epoch"] = s.get("best_rec_epoch")
            print(f"  {name:<16} train {tr:.6f} · validation {va:.6f}", flush=True)
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": repr(e)}
            print(f"  {name:<16} 실패: {e!r}", flush=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default=str(REPO / "data/output/fixedcluster_v1"))
    ap.add_argument("--v2", default=None, help="기본 <pre>/v2")
    ap.add_argument("--ym", default=str(Path.home() / "minji/yeomin_code_22028ba"))
    ap.add_argument("--v1-sid", default=str(Path.home() / "minji/uni_post_rqvae/article_semantic_ids.parquet"))
    ap.add_argument("--no-rec", action="store_true")
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    pre = Path(a.pre).expanduser()
    v2 = Path(a.v2).expanduser() if a.v2 else pre / "v2"
    inp = pre / "exports" / "rqvae_inputs"
    res = {"inputs": {"v2": str(v2), "rqvae_inputs": str(inp)}, "sid": {}, "reconstruction": {}}

    runs = {"V1": Path(a.v1_sid).expanduser(), "H": v2 / "sid/h/article_semantic_ids.parquet",
            "R": v2 / "sid/r/article_semantic_ids.parquet"}
    rows = []
    for run, path in runs.items():
        if not path.exists():
            print(f"[{run}] SID 파일 없음: {path}")
            continue
        sid = load_sid(path, inp)
        res["sid"][run] = {}
        for split in ["train", "validation", "all"]:
            part = sid if split == "all" else sid[sid.split == split]
            if not len(part):
                continue
            st = sid_stats(part)
            res["sid"][run][split] = st
            rows.append({"실험": run, "split": split, **st})
    if rows:
        t = pd.DataFrame(rows)
        for c in ["충돌률", "c4>0 비율"]:
            t[c] = (t[c] * 100).map(lambda v: f"{v:.2f}%")
        print("\n== SID 충돌 · 코드 사용 (코드북 크기 c1 25 · c2 128 · c3 512)")
        print("   V1 SID 파일에 split 열이 없으면 'all'만 나온다")
        print(t.to_string(index=False))

    if not a.no_rec:
        ckpts = sorted((v2 / "rqvae").glob("*/checkpoint_best_rec.pt"))
        print(f"\n== 재구성 손실 (기사 평균, checkpoint_best_rec · {len(ckpts)}개)")
        res["reconstruction"] = rec_losses(Path(a.ym).expanduser(), inp, ckpts, a.batch)
        print("   V1(예전 UNI) checkpoint 는 예전 전처리 입력이라 여기서 재지 않는다")

    out = v2 / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    f, k = out / "sid_quality.json", 2
    while f.exists():
        f, k = out / f"sid_quality_{k}.json", k + 1
    f.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {f}")


if __name__ == "__main__":
    main()
