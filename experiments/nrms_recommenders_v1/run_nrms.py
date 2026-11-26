"""Microsoft Recommenders NRMS (모델 코드 수정 없음) 를 prepare_mind.py 결과로 학습 + test.

    python experiments/nrms_recommenders_v1/run_nrms.py --data <prepare_mind 폴더> --out <결과 폴더> --seed 42 [--lr 1e-4 --dropout 0.2 ...]

모델 · 학습 단계 · 점수: recommenders.models.newsrec (NRMSModel, MINDIterator) 그대로.
  학습 = 공개 코드의 train_iterator 로 1 epoch 씩 model.train(batch) (BaseModel.fit 과 같은 반복), 정답 1 + 오답 4 softmax.
  평가 = 공개 코드의 run_fast_eval (뉴스 벡터 · 사용자 벡터 내적). 우리 지표 (popularity_fusion.metrics, 동점 기대값) 로 계산.
  epoch 마다 validation 절반 Top-1 → 가장 좋은 epoch 의 가중치로 test 한 번.
  공개 코드의 test_iterator 는 처음 읽은 노출 파일을 기억하므로, test 를 평가하기 전에 반복자를 새로 만든다.
결과 (폴더가 이미 있으면 멈춤): config.json, epoch_history.csv, best_weights.npz, {val,test}_candidate_scores.parquet, test_metrics.json
  점수 파일 열은 Transformer 예측과 같다 (sample_index, impression_id, user_id, impression_time, article_id, c1, c2, label, candidate_score)
  → popularity_window.py --pred-template 으로 NRMS + 인기 + 나이를 계산할 수 있다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2_yeomin_rerun"))
from popularity_fusion import metrics  # noqa: E402

HERE = Path(__file__).resolve().parent


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scores(model, news, beh, n):
    impr, _labels, preds = model.run_fast_eval(str(news), str(beh))
    S = np.full((n, 5), np.nan)
    for i, p in zip(impr, preds):
        S[i] = p
    if np.isnan(S).any():
        raise ValueError(f"{beh.name}: 점수가 빠진 행이 있습니다")
    return S


def to_frame(meta: pd.DataFrame, S: np.ndarray) -> pd.DataFrame:
    n = len(meta)
    return pd.DataFrame({"sample_index": np.repeat(np.arange(n), 5),
                         "impression_id": np.repeat(meta.impression_id.to_numpy(), 5),
                         "user_id": np.repeat(meta.user_id.to_numpy(), 5),
                         "impression_time": np.repeat(meta.impression_time.to_numpy(), 5),
                         "article_id": np.concatenate(meta.candidate_article_ids.to_numpy()),
                         "c1": np.concatenate(meta.candidate_c1.to_numpy()), "c2": np.concatenate(meta.candidate_c2.to_numpy()),
                         "label": np.concatenate(meta.candidate_labels.to_numpy()), "candidate_score": S.reshape(-1)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--yaml", default=str(HERE / "nrms.yaml"), help="기본값 = 공개 코드 MIND 예제 설정")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--head-num", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    data, out = Path(a.data).expanduser(), Path(a.out).expanduser()
    if out.exists():
        sys.exit(f"멈춤: {out} 가 이미 있습니다 (덮어쓰지 않음)")

    import tensorflow as tf
    from recommenders.models.newsrec.io.mind_iterator import MINDIterator
    from recommenders.models.newsrec.models.nrms import NRMSModel
    from recommenders.models.newsrec.newsrec_utils import prepare_hparams

    over = {k: v for k, v in {"learning_rate": a.lr, "dropout": a.dropout, "head_num": a.head_num, "head_dim": a.head_dim,
                              "batch_size": a.batch_size, "epochs": a.epochs}.items() if v is not None}
    hp = prepare_hparams(a.yaml, wordEmb_file=str(data / "embedding.npy"), wordDict_file=str(data / "word_dict.pkl"),
                         userDict_file=str(data / "uid2index.pkl"), **over)
    out.mkdir(parents=True)
    cfg = {"seed": a.seed, "data": str(data), "overrides": over,
           **{k: getattr(hp, k) for k in ("title_size", "his_size", "npratio", "word_emb_dim", "dropout", "attention_hidden_dim",
                                           "head_num", "head_dim", "learning_rate", "optimizer", "loss", "epochs", "batch_size")}}
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    log("설정 " + " · ".join(f"{k} {v}" for k, v in cfg.items() if k not in ("data",)) + f" · TF {tf.__version__} · GPU {tf.config.list_physical_devices('GPU')}")

    news = data / "news.tsv"
    meta = {k: pd.read_parquet(data / f"{k}_meta.parquet") for k in ("valid", "test")}
    Y = {k: np.stack(meta[k].candidate_labels.to_numpy()) for k in meta}
    def met(S, y):
        return metrics(pd.DataFrame({"label": y.reshape(-1)}), S.reshape(-1))

    model = NRMSModel(hp, MINDIterator, seed=a.seed)
    hist, best, best_ep, best_w = [], -1.0, 0, None
    S0 = scores(model, news, data / "valid_behaviors.tsv", len(meta["valid"]))
    log(f"[epoch 0] val Top-1 {met(S0, Y['valid'])['top1']:.2%}")
    for ep in range(1, hp.epochs + 1):
        t0, tot, n = time.time(), 0.0, 0
        for step, batch in enumerate(model.train_iterator.load_data_from_file(str(news), str(data / "train_behaviors.tsv")), 1):
            loss = model.train(batch)
            tot += float(np.asarray(loss).reshape(-1)[0]); n += 1
            if step % 1000 == 0:
                log(f"  [Train {ep}/{hp.epochs}] step {step:,} | loss {tot / n:.4f} | {time.time() - t0:.0f}s")
        S = scores(model, news, data / "valid_behaviors.tsv", len(meta["valid"]))
        m = met(S, Y["valid"])
        hist.append({"epoch": ep, "train_loss": tot / max(n, 1), **{f"val_{k}": v for k, v in m.items()}, "sec": time.time() - t0})
        pd.DataFrame(hist).to_csv(out / "epoch_history.csv", index=False)
        log(f"Epoch {ep} | train loss {tot / max(n, 1):.4f} | val Top-1 {m['top1']:.2%} · AUC {m['auc']:.4f} · MRR {m['mrr']:.4f} | {time.time() - t0:.0f}s")
        if m["top1"] > best + 1e-12:
            best, best_ep = m["top1"], ep
            best_w = model.model.get_weights()      # 공개 코드가 TF1 세션(그래프) 방식이라 save_weights 대신 메모리에 보관
            log(f"  ✓ Best (val Top-1 {best:.2%})")

    model.model.set_weights(best_w)
    np.savez_compressed(out / "best_weights.npz", *best_w)
    res = {"best_epoch": best_ep, "config": cfg}
    for k in ("valid", "test"):
        model.test_iterator = MINDIterator(hp, col_spliter="\t")      # 기억해 둔 노출 파일을 버리고 새로 읽게
        S = scores(model, news, data / f"{k}_behaviors.tsv", len(meta[k]))
        to_frame(meta[k], S).to_parquet(out / f"{'val' if k == 'valid' else 'test'}_candidate_scores.parquet", index=False)
        res["validation" if k == "valid" else "test"] = met(S, Y[k])
    res["test_rows"] = int(len(meta["test"]))
    (out / "test_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    t = res["test"]
    log(f"=== Test (best epoch {best_ep}, {len(meta['test']):,}행) Top-1 {t['top1']:.2%} · AUC {t['auc']:.4f} · MRR {t['mrr']:.4f} · nDCG@5 {t['ndcg@5']:.4f}")
    if abs(res["validation"]["top1"] - best) > 1e-9:
        log(f"주의: 다시 계산한 validation Top-1 {res['validation']['top1']:.4%} ≠ 학습 중 {best:.4%}")


if __name__ == "__main__":
    main()
