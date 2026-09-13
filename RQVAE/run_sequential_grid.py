"""
MIND RQ-VAE 순차(coordinate-wise) 하이퍼파라미터 튜닝 스크립트
민지님 event_parameter_sweep 패턴을 따름 (전용 폴더, 절대 덮어쓰지 않음, 요약 자동 취합)

=== 사용법 ===
1. 아래 CURRENT_BEST를 "지금까지 확정된 최적값"으로 채워둔다.
2. AXIS_TO_SWEEP을 이번에 스윕할 축 하나로 지정한다.
   (latent_dim -> q3 -> quantizer -> normalize -> lambda_cb -> lambda_com 순서로 진행)
3. 실행: !python run_sequential_grid.py
4. 결과(grid_summary.csv) 보고 이번 축의 승자를 고른다.
5. CURRENT_BEST에 그 값을 반영하고, AXIS_TO_SWEEP을 다음 축으로 바꿔서 다시 실행.

세션이 끊기면: !python run_sequential_grid.py --skip-existing
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/home/ubuntu/mingyo/SID_Project/RQVAE")
GIN_PATH = PROJECT_ROOT / "configs" / "rqvae_mind.gin"
GRID_ROOT = PROJECT_ROOT / "out" / "rqvae" / "mind_sequential_grid"

# ============================================================
# 여기 두 개만 매 단계마다 손으로 바꾸면 됨
# ============================================================

# 지금까지 확정된 최적값 (처음엔 TIGER/기존 기본값으로 시작)
CURRENT_BEST: dict[str, Any] = {
    "embed_dim": 128,            # latent_dim
    "c3": 512,                   # Q3 size
    "quantizer": "STE",  # STE / GUMBEL_SOFTMAX / ROTATION_TRICK
    "normalize": False,          # True / False
    "lambda_cb": 1.0,
    "lambda_com": 1,
}

# 이번에 스윕할 축 하나 (아래 AXES 딕셔너리의 key 중 하나)
AXIS_TO_SWEEP = "lambda_com"

# ============================================================
# 고정값 (이미 확정, 절대 안 바뀜)
# ============================================================
FIXED_C2 = 256
FIXED_LAMBDA_UNIQ = 0.1
FIXED_MARGIN = 0.5

# ============================================================
# 각 축의 gin 키 / 후보값 / 값 포맷 함수
# ============================================================

def _fmt_plain(v: Any) -> str:
    return str(v)


def _fmt_enum(v: Any) -> str:
    return f"%QuantizeForwardMode.{v}"


AXES: dict[str, dict[str, Any]] = {
    "latent_dim": {
        "gin_key": "vae_embed_dim",
        "values": [32, 64, 128, 256, 512],
        "fmt": _fmt_plain,
        "best_key": "embed_dim",
    },
    "q3": {
        "gin_key": "vae_c3_codebook_size",
        "values": [128, 256, 512, 1024],
        "fmt": _fmt_plain,
        "best_key": "c3",
    },
    "quantizer": {
        "gin_key": "vae_codebook_mode",
        "values": ["STE", "GUMBEL_SOFTMAX", "ROTATION_TRICK"],
        "fmt": _fmt_enum,
        "best_key": "quantizer",
    },
    "normalize": {
        "gin_key": "vae_codebook_normalize",
        "values": [True, False],
        "fmt": _fmt_plain,
        "best_key": "normalize",
    },
    "lambda_cb": {
        "gin_key": "lambda_cb",
        "values": [0.25, 0.5, 1, 2],
        "fmt": _fmt_plain,
        "best_key": "lambda_cb",
    },
    "lambda_com": {
        "gin_key": "lambda_com",
        "values": [0.1, 0.25, 1, 2],
        "fmt": _fmt_plain,
        "best_key": "lambda_com",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _slug(v: Any) -> str:
    return str(v).replace(".", "p").replace(" ", "")


def _run_name(axis: str, value: Any) -> str:
    fixed_part = "_".join(
        f"{k}{_slug(v)}" for k, v in CURRENT_BEST.items() if AXES[axis]["best_key"] != k
    )
    return f"{axis}_{_slug(value)}__{fixed_part}"


def _set_gin_value(content: str, key: str, formatted_value: str) -> str:
    pattern = rf"train\.{key} = [^\n]+"
    if re.search(pattern, content) is None:
        raise ValueError(f"gin 파일에서 train.{key} 를 찾지 못했습니다.")
    return re.sub(pattern, f"train.{key} = {formatted_value}", content)


def _write_gin_config(axis: str, value: Any, save_dir: Path) -> None:
    with open(GIN_PATH, "r") as f:
        content = f.read()

    # 1) 고정값 (c2, lambda_uniq, margin)
    content = _set_gin_value(content, "vae_c2_codebook_size", str(FIXED_C2))
    content = _set_gin_value(content, "lambda_uniq", str(FIXED_LAMBDA_UNIQ))
    content = _set_gin_value(content, "uniqueness_margin", str(FIXED_MARGIN))

    # 2) CURRENT_BEST 값들 전부 반영 (이번에 스윕하는 축만 value로 덮어씀)
    for axis_name, cfg in AXES.items():
        best_key = cfg["best_key"]
        v = value if axis_name == axis else CURRENT_BEST[best_key]
        content = _set_gin_value(content, cfg["gin_key"], cfg["fmt"](v))

    # 3) 저장 경로
    content = re.sub(
        r'train\.save_dir_root = "[^"]*"',
        f'train.save_dir_root = "{save_dir.relative_to(PROJECT_ROOT)}/"',
        content,
    )

    with open(GIN_PATH, "w") as f:
        f.write(content)


def _run_worker(axis: str, value: Any, run_dir: Path) -> dict[str, Any]:
    _write_gin_config(axis, value, run_dir)

    subprocess.run(
        [sys.executable, "train_rqvae.py", "configs/rqvae_mind.gin"],
        cwd=PROJECT_ROOT,
        check=True,
    )

    sid_out = run_dir / "semantic_ids"
    subprocess.run(
        [
            sys.executable, "generate_semantic_ids.py",
            "--data_dir", "datasets/mind",
            "--checkpoint", str(run_dir / "checkpoint_final.pt"),
            "--output_dir", str(sid_out),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    eval_result = subprocess.run(
        [
            sys.executable, "evaluate/eval_sid_collision.py",
            "--sid", str(sid_out / "article_semantic_ids.parquet"),
            "--data-dir", "datasets/mind",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    output = eval_result.stdout
    print(output)

    rate_match = re.search(r"Collision Rate\s*:\s*([\d.]+)%", output)
    unique_match = re.search(r"Unique c123\s*:\s*(\d+)", output)

    row = {
        "axis": axis,
        "swept_value": value,
        "collision_rate_pct": float(rate_match.group(1)) if rate_match else None,
        "unique_c123": int(unique_match.group(1)) if unique_match else None,
        "run_dir": str(run_dir),
    }
    row.update({f"fixed_{k}": v for k, v in CURRENT_BEST.items()})
    row["fixed_c2"] = FIXED_C2
    row["fixed_lambda_uniq"] = FIXED_LAMBDA_UNIQ
    row["fixed_margin"] = FIXED_MARGIN
    return row


def _write_grid_summary(axis: str, rows: list[dict[str, Any]]) -> None:
    axis_root = GRID_ROOT / axis
    axis_root.mkdir(parents=True, exist_ok=True)

    (axis_root / "grid_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if rows:
        fieldnames = list(rows[0].keys())
        with (axis_root / "grid_summary.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def main() -> None:
    args = _parse_args()

    axis = AXIS_TO_SWEEP
    cfg = AXES[axis]
    values = cfg["values"]

    axis_root = GRID_ROOT / axis
    axis_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"MIND RQ-VAE 순차 튜닝 — 이번 축: {axis}")
    print("=" * 80)
    print(f"후보값: {values}")
    print(f"고정값(다른 축들): {CURRENT_BEST}")
    print(f"공통 고정: c2={FIXED_C2}, lambda_uniq={FIXED_LAMBDA_UNIQ}, margin={FIXED_MARGIN}")
    print("Overwrite: NEVER")
    print("=" * 80)

    summary_rows: list[dict[str, Any]] = []

    for value in values:
        name = _run_name(axis, value)
        run_dir = axis_root / name
        complete_marker = run_dir / "_COMPLETE.json"

        if run_dir.exists():
            if complete_marker.exists() and args.skip_existing:
                print(f"[SKIP] {name} (이미 완료됨)")
                row = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            else:
                raise FileExistsError(
                    "동일 조합 폴더가 이미 존재하므로 절대 덮어쓰지 않습니다.\n"
                    f"경로={run_dir}\n"
                    "완료된 조합을 건너뛰려면 --skip-existing를 사용하세요."
                )
        else:
            print()
            print("#" * 80)
            print(f"[RUN] {name}")
            print("#" * 80)

            run_dir.mkdir(parents=True, exist_ok=False)

            try:
                row = _run_worker(axis, value, run_dir)
            except Exception:
                print(f"[FAILED] 부분 결과를 진단용으로 보존합니다: {run_dir}")
                raise

            (run_dir / "run_summary.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (run_dir / "_COMPLETE.json").write_text(
                json.dumps({"status": "SUCCESS", **row}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        summary_rows.append(row)
        _write_grid_summary(axis, summary_rows)

    print()
    print("=" * 80)
    print(f"'{axis}' 축 스윕 완료 ({len(values)}개)")
    print(f"CSV : {axis_root / 'grid_summary.csv'}")
    print("=" * 80)
    print("\n이 결과 보고 최적값 고른 다음, CURRENT_BEST 업데이트하고")
    print("AXIS_TO_SWEEP을 다음 축으로 바꿔서 다시 실행하세요.")


if __name__ == "__main__":
    main()
