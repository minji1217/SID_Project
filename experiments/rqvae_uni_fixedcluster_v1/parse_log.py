"""train.log 에서 epoch별 학습 지표와 validation 지표를 CSV로 뽑는다 (학습 중에도 실행 가능).

train_rqvae.py 는 지표를 화면(로그)에만 찍는다 (wandb 꺼짐). 그래서 로그를 표로 바꿔 남긴다.

출력 (OUT 폴더)
  metrics_train.csv       epoch, loss, rec, cb, com, uniq, rqvae, t, global_step       (매 epoch)
  metrics_validation.csv  epoch, loss, rec, cb, com, uniq, rqvae                       (평가한 epoch, 10 epoch마다)
  early_stopping.csv      epoch, improved_total, improved_rec, counter, best_total, best_total_epoch, best_rec, best_rec_epoch
  summary.json            best_rec / best_total epoch 과 값, 마지막 epoch, 조기 종료 여부

사용: python parse_log.py <OUT 폴더>
"""
import csv
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
text = (out / "train.log").read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
kv = lambda s: {k: float(v) for k, v in re.findall(r"(\w+)=([-+0-9.eE]+)", s)}

train_rows, val_rows, es_rows = [], [], []
for line in text.splitlines():
    m = re.match(r"^\[Epoch\s+(\d+)/\d+\]\s*(.*)$", line)
    if m:
        train_rows.append({"epoch": int(m.group(1)), **kv(m.group(2))})
        continue
    m = re.match(r"^\[Validation epoch (\d+)\]\s*(.*)$", line)
    if m:
        val_rows.append({"epoch": int(m.group(1)), **kv(m.group(2))})
        continue
    m = re.match(r"^\[EarlyStopping\] epoch=(\d+) \| improved_total=(\w+) \| improved_rec=(\w+) \| counter=(\d+)/\d+ \| "
                 r"best_total=([-0-9.eE]+)@(\d+) \| best_rec=([-0-9.eE]+)@(\d+)", line)
    if m:
        es_rows.append({"epoch": int(m.group(1)), "improved_total": m.group(2), "improved_rec": m.group(3),
                        "counter": int(m.group(4)), "best_total": float(m.group(5)), "best_total_epoch": int(m.group(6)),
                        "best_rec": float(m.group(7)), "best_rec_epoch": int(m.group(8))})


def write(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


write("metrics_train.csv", train_rows)
write("metrics_validation.csv", val_rows)
write("early_stopping.csv", es_rows)

last_es = es_rows[-1] if es_rows else {}
summary = {
    "train_epochs_logged": len(train_rows),
    "last_epoch": train_rows[-1]["epoch"] if train_rows else None,
    "validation_evals": len(val_rows),
    "best_rec": last_es.get("best_rec"),
    "best_rec_epoch": last_es.get("best_rec_epoch"),
    "best_total": last_es.get("best_total"),
    "best_total_epoch": last_es.get("best_total_epoch"),
    "early_stopped": bool(re.search(r"Early stopped: True", text)),
    "finished": "Final model saved" in text,
}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
