"""nrms_epochs20_ebnerd_v1 결과 요약: 모드별 seed 평균 + 5 epoch 결과(v2 / recon_v1)와 비교. 읽기만 한다."""
import glob, json, re, statistics as st, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data/output/experiments/nrms_epochs20_ebnerd_v1"
FIVE = {"e5": 29.92, "recon": 29.99, "sid": 26.09, "sid_shuffled": 20.71, "title": 26.06}

print(f"{'mode':13s} {'n':>2s} {'val':>7s} {'test':>7s} {'std':>5s} {'MRR':>7s} {'AUC':>7s} {'5ep test':>9s}  best epoch / seed")
out = {}
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    js = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/metrics_*_s*.json"))]
    if not js:
        continue
    t = [j["test"]["top1_accuracy"] * 100 for j in js]
    best = []
    for f in sorted(glob.glob(f"{d}/log_*_s*.txt")):
        v = [float(x) for x in re.findall(r"val_top1=([\d.]+)%", open(f).read())]
        best.append(f"{re.search(r'_s(\d+)\.txt', f).group(1)}:{v.index(max(v)) + 1}/{len(v)}" if v else "?")
    m = lambda k: st.mean(j["test"][k] for j in js)
    row = {"seeds": len(js), "validation_top1": st.mean(j["validation"]["top1_accuracy"] for j in js) * 100,
           "test_top1": st.mean(t), "test_top1_std": st.stdev(t) if len(t) > 1 else 0.0,
           "test_mrr": m("mrr"), "test_ndcg@5": m("ndcg@5"), "test_auc": m("auc"),
           "test_top1_per_seed": t, "best_epoch": best}
    out[d.name] = row
    print(f"{d.name:13s} {len(js):2d} {row['validation_top1']:6.2f}% {row['test_top1']:6.2f}% {row['test_top1_std']:5.2f} "
          f"{row['test_mrr']:.4f} {row['test_auc']:.4f} {FIVE.get(d.name, float('nan')):8.2f}%  {' '.join(best)}")
json.dump(out, open(root / "summary.json", "w"), indent=2, ensure_ascii=False)
print(f"\n저장: {root / 'summary.json'}")
