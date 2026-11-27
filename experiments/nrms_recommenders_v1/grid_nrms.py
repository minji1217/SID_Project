"""NRMS 그리드서치 (교수님 방향: 공개 코드 기본값에서 출발해 조금씩 바꿔 봄) → 가장 좋은 설정으로 seed 3개.

    python experiments/nrms_recommenders_v1/grid_nrms.py --data <prepare_mind 폴더> --root <결과 폴더>

1) seed 42 로 학습률 × dropout 격자 (기본 1e-4 · 0.2 를 가운데에 두고 한 칸씩): run_nrms.py 를 차례로 (--parallel 로 동시에) 실행
2) validation 절반 Top-1 이 가장 높은 설정 (같으면 기본값에 가까운 쪽) → seed 123 · 2026 으로 같은 설정 실행
3) <root>/summary.csv · summary.json: 격자 전체 (validation · test) 와 최종 3-seed 평균. test 는 고르는 데 쓰지 않는다.
   <root>/final/best_seed<seed>/ = 최종 설정의 seed 별 결과 (링크). NRMS + 인기 + 나이:
   popularity_window.py --pred-template '{v2}/nrms/{run}/final/best_seed{seed}/{split}_candidate_scores.parquet' --out-name nrms_window
이미 끝난 실행 (test_metrics.json 있음) 은 건너뛴다. 끝나지 않은 폴더가 있으면 멈춘다 (지우고 다시).
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT = {"lr": 1e-4, "dropout": 0.2}


def name(lr, do, seed):
    return f"lr{lr:g}_do{do:g}_seed{seed}"


def run(py, data, out: Path, lr, do, seed, extra):
    if (out / "test_metrics.json").exists():
        print(f"[건너뜀] {out.name} (이미 끝남)", flush=True)
        return
    if out.exists():
        sys.exit(f"멈춤: {out} 가 끝나지 않은 채 있습니다. 확인 후 지우고 다시 실행하세요")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [py, str(HERE / "run_nrms.py"), "--data", str(data), "--out", str(out), "--seed", str(seed),
           "--lr", str(lr), "--dropout", str(do), *extra]
    print(f"[{time.strftime('%H:%M:%S')}] 시작 {out.name}", flush=True)
    with open(str(out) + ".log", "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    print(f"[{time.strftime('%H:%M:%S')}] {'끝' if rc == 0 else f'실패 (exit {rc}), 로그: {out}.log'} {out.name}", flush=True)


def read(out: Path):
    r = json.loads((out / "test_metrics.json").read_text())
    return {"best_epoch": r["best_epoch"], **{f"val_{k}": v for k, v in r["validation"].items()}, **{f"test_{k}": v for k, v in r["test"].items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--lrs", nargs="+", type=float, default=[5e-5, 1e-4, 2e-4])
    ap.add_argument("--dropouts", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    ap.add_argument("--grid-seed", type=int, default=42)
    ap.add_argument("--final-seeds", nargs="+", type=int, default=[123, 2026])
    ap.add_argument("--parallel", type=int, default=1, help="동시에 돌릴 실행 수 (GPU 하나에 2 정도)")
    ap.add_argument("--epochs", type=int, default=None, help="(테스트용) 기본 5")
    a = ap.parse_args()
    data, root = Path(a.data).expanduser(), Path(a.root).expanduser()
    extra = ["--epochs", str(a.epochs)] if a.epochs else []

    grid = [(lr, do) for lr in a.lrs for do in a.dropouts]
    print(f"격자 {len(grid)}개 (seed {a.grid_seed}): " + ", ".join(name(lr, do, a.grid_seed) for lr, do in grid), flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(lambda c: run(a.python, data, root / "grid" / name(*c, a.grid_seed), *c, a.grid_seed, extra), grid))
    rows = []
    for lr, do in grid:
        out = root / "grid" / name(lr, do, a.grid_seed)
        if (out / "test_metrics.json").exists():
            rows.append({"lr": lr, "dropout": do, "seed": a.grid_seed, **read(out)})
    if not rows:
        sys.exit("끝난 격자 실행이 없습니다")
    G = pd.DataFrame(rows)
    dist = lambda r: abs(np.log(r.lr / DEFAULT["lr"])) + abs(r.dropout - DEFAULT["dropout"])
    G["_d"] = G.apply(dist, axis=1)
    best = G.sort_values(["val_top1", "_d"], ascending=[False, True]).iloc[0]
    lr, do = float(best.lr), float(best.dropout)
    print(f"고른 설정 (validation Top-1 최고): lr {lr:g} · dropout {do:g} → val {best.val_top1:.2%}", flush=True)

    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(lambda s: run(a.python, data, root / "final" / name(lr, do, s), lr, do, s, extra), a.final_seeds))
    # 최종 설정의 seed 별 결과를 같은 이름 규칙으로: final/best_seed<seed> → 실제 폴더 (seed 42 는 grid/ 에 있음)
    (root / "final").mkdir(parents=True, exist_ok=True)
    for s_, tgt in [(a.grid_seed, Path("..") / "grid" / name(lr, do, a.grid_seed))] + [(s_, Path(name(lr, do, s_))) for s_ in a.final_seeds]:
        link = root / "final" / f"best_seed{s_}"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(tgt, target_is_directory=True)
    final = [{"seed": a.grid_seed, **read(root / "grid" / name(lr, do, a.grid_seed))}]
    for s in a.final_seeds:
        out = root / "final" / name(lr, do, s)
        if (out / "test_metrics.json").exists():
            final.append({"seed": s, **read(out)})
    F = pd.DataFrame(final)
    G.drop(columns="_d").assign(구분="격자").to_csv(root / "summary.csv", index=False)
    with open(root / "summary.csv", "a") as f:
        F.assign(lr=lr, dropout=do, 구분="최종").to_csv(f, index=False, header=False, columns=list(G.drop(columns="_d").columns) + ["구분"])
    summ = {"chosen": {"lr": lr, "dropout": do}, "grid": G.drop(columns="_d").to_dict("records"), "final": final,
            "final_mean": {k: float(F[k].mean()) for k in F.columns if k.startswith(("val_", "test_"))}}
    (root / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n격자 (seed %d):" % a.grid_seed)
    print(G.drop(columns="_d")[["lr", "dropout", "best_epoch", "val_top1", "test_top1", "test_auc"]].to_string(index=False))
    print(f"\n최종 lr {lr:g} · dropout {do:g}, seed {', '.join(str(x['seed']) for x in final)}: test Top-1 "
          + " / ".join(f"{x['test_top1']:.2%}" for x in final) + f" → 평균 {F.test_top1.mean():.2%} · AUC {F.test_auc.mean():.4f} · MRR {F.test_mrr.mean():.4f}")
    print(f"저장: {root}/summary.csv, summary.json")


if __name__ == "__main__":
    main()
