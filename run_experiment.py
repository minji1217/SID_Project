from __future__ import annotations

"""
RQ-VAE staged hyperparameter experiment runner.

이 스크립트의 목적
------------------
1) 한 번의 명령으로 한 Stage의 모든 조합을 순차 실행한다.
2) 각 실험마다 원본 gin 파일을 수정하지 않고 별도 config.gin을 만든다.
3) RQ-VAE 학습 -> Semantic ID(c1,c2,c3,c4) 생성 -> 전체 평가를 자동 수행한다.
4) 실험 도중 중단되어도 완료된 EXP는 건너뛰고 이어서 실행한다.
5) Stage가 끝나면 stage_results.csv / stage_report.html / all_results.csv를 만든다.

중요한 설계 원칙
----------------
- 이 스크립트는 "자동으로 1등을 확정"하지 않는다.
  여러 지표(C2 Event Consistency, Collision, Rec Loss, Semantic Similarity, Utilization)의
  trade-off를 사람이 확인한 뒤 다음 Stage의 parent EXP를 선택하도록 설계했다.
- 기본 실행은 GPU 하나에서 EXP를 하나씩 순차 실행한다.
  "16개를 한 번에 돌린다"는 의미는 명령을 한 번만 입력하면 16개가 자동으로 이어서
  실행된다는 뜻이며, 16개 학습을 동시에 GPU에 올린다는 뜻이 아니다.
- 최신 Semantic ID는 c1,c2,c3,c4를 사용한다. 다만 Collision Rate 평가는 의도적으로
  c1,c2,c3 충돌을 측정한다. c4는 바로 그 충돌을 최종적으로 구분하기 위한 코드이기 때문이다.

권장 위치
---------
SID_Project-main/RQVAE/run_experiment.py

대표 실행 예시
-------------
# 1) 계획만 확인
python run_experiment.py --stage q2q3 --dataset ebnerd --dry-run

# 2) 첫 조합 하나만 smoke test
python run_experiment.py --stage q2q3 --dataset ebnerd --limit 1

# 3) Q2 x Q3 16개 전체 실행
python run_experiment.py --stage q2q3 --dataset ebnerd

# 4) Stage 1 결과에서 선택한 TOP 3를 parent로 Latent 탐색
python run_experiment.py --stage latent --dataset ebnerd \
    --parents Q2Q3-002-a1b2c3,Q2Q3-006-d4e5f6,Q2Q3-011-123abc

Stage가 끝난 뒤 stage_results.csv를 팀에서 검토하거나 ChatGPT에 전달하고,
선정한 EXP ID를 다음 Stage의 --parents에 넣으면 된다.
"""

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


# ============================================================================
# 기본 설정
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_CONFIGS = {
    "ebnerd": Path("configs/rqvae_ebnerd.gin"),
    "mind": Path("configs/rqvae_mind.gin"),
}

# Stage 결과 CSV에 함께 남길 주요 gin 파라미터.
# "이번 Stage에서 무엇만 바뀌었는지"뿐 아니라 현재 모델 전체 핵심 설정도 한 행에서 확인할 수 있다.
TRACKED_CONFIG_KEYS = [
    "train.epochs",
    "train.batch_size",
    "train.learning_rate",
    "train.weight_decay",
    "train.gradient_accumulate_every",
    "train.dataset_folder",
    "train.vae_input_dim",
    "train.vae_hidden_dims",
    "train.vae_embed_dim",
    "train.vae_num_categories",
    "train.vae_c2_codebook_size",
    "train.vae_c3_codebook_size",
    "train.vae_codebook_normalize",
    "train.vae_sim_vq",
    "train.vae_codebook_mode",
    "train.lambda_rec",
    "train.lambda_cb",
    "train.lambda_com",
    "train.kmeans_encode_batch_size",
    "train.kmeans_n_init",
    "train.gumbel_t0",
    "train.gumbel_min_t",
    "train.gumbel_anneal_rate",
    "train.eval_every_epochs",
    "train.seed",
]

# 평가 로그에서 수집하는 지표 이름. CSV 컬럼 순서를 일정하게 유지하기 위해 별도 정의한다.
METRIC_COLUMNS = [
    "train_rec_loss",
    "valid_rec_loss",
    "c2_event_consistency",
    "evaluated_multi_article_events",
    "total_articles",
    "q2_used_codes",
    "q2_utilization",
    "q3_used_codes",
    "q3_utilization",
    "unique_c123",
    "collision_articles",
    "collision_rate",
    "delta_c2_similarity",
    "delta_c3_similarity",
    "c1_category_accuracy",
    "final_sid_uniqueness",
    "max_c4",
    "best_valid_total_loss",
    "best_valid_total_epoch",
    "best_valid_rec_loss",
    "best_valid_rec_epoch",
    "final_valid_total_loss",
    "final_valid_rec_loss",
]

# Quantizer 이름 -> gin enum 표현식.
QUANTIZER_GIN = {
    "STE": "%QuantizeForwardMode.STE",
    "GUMBEL": "%QuantizeForwardMode.GUMBEL_SOFTMAX",
    "ROTATION": "%QuantizeForwardMode.ROTATION_TRICK",
}


@dataclass(frozen=True)
class ExperimentPlan:
    """Stage 안에서 실행할 실험 하나의 계획."""

    exp_id: str
    stage: str
    parent_exp_id: str | None
    base_config_path: Path
    overrides: dict[str, Any]
    exp_dir: Path


# ============================================================================
# 공통 유틸리티
# ============================================================================


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_dump(path: Path, payload: Any) -> None:
    """상태 파일은 중간에 깨지지 않도록 임시 파일에 쓴 뒤 교체한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def parse_scalar(text: str) -> Any:
    """
    CLI --values 값을 Python 값으로 변환한다.

    예)
    128 -> int
    0.25 -> float
    True -> bool
    '[512, 256]' -> list
    STE -> str
    """

    text = text.strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_values_csv(text: str | None) -> list[Any] | None:
    """
    간단한 comma separated 값 파서.

    리스트 자체에 comma가 들어가는 hidden_dims 같은 값은 preset Stage를 사용하거나
    custom Stage에서 --values '[512];[512,256]'처럼 semicolon을 사용한다.
    """

    if text is None:
        return None
    separator = ";" if ";" in text else ","
    values = [parse_scalar(item) for item in text.split(separator) if item.strip()]
    if not values:
        raise ValueError("--values에 최소 1개 값이 필요합니다.")
    return values


def read_config(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"gin config를 찾을 수 없습니다: {path}")
    return path.read_text(encoding="utf-8")


def extract_config_value(config_text: str, key: str) -> str | None:
    """gin 파일에서 'train.foo = value'의 value 부분을 문자열로 읽는다."""

    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(.*?)\s*(?:#.*)?$", re.MULTILINE)
    match = pattern.search(config_text)
    return match.group(1).strip() if match else None


def gin_value(value: Any) -> str:
    """Python 값을 gin 파일에 쓸 표현식으로 변환한다."""

    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        # Quantizer enum은 문자열 리터럴이 아니라 gin enum 참조로 써야 한다.
        if value in QUANTIZER_GIN:
            return QUANTIZER_GIN[value]
        if value.startswith("%QuantizeForwardMode."):
            return value
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return repr(list(value))
    if value is None:
        return "None"
    return repr(value)


def set_config_values(config_text: str, overrides: dict[str, Any]) -> str:
    """
    원본 gin 텍스트에서 필요한 키만 교체한다.

    기존 키가 없는 경우 파일 맨 아래에 추가한다. 원본 파일 자체는 절대 수정하지 않고,
    각 EXP 폴더의 config.gin에 결과를 저장한다.
    """

    result = config_text
    for key, value in overrides.items():
        replacement = f"{key} = {gin_value(value)}"
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=.*$", re.MULTILINE)
        if pattern.search(result):
            result = pattern.sub(replacement, result, count=1)
        else:
            result = result.rstrip() + "\n" + replacement + "\n"
    return result


def config_snapshot(config_text: str) -> dict[str, str | None]:
    return {key: extract_config_value(config_text, key) for key in TRACKED_CONFIG_KEYS}


def config_int(config_text: str, key: str) -> int:
    value = extract_config_value(config_text, key)
    if value is None:
        raise KeyError(f"config에 {key}가 없습니다.")
    return int(ast.literal_eval(value))


def config_string(config_text: str, key: str) -> str:
    value = extract_config_value(config_text, key)
    if value is None:
        raise KeyError(f"config에 {key}가 없습니다.")
    try:
        parsed = ast.literal_eval(value)
        return str(parsed)
    except (ValueError, SyntaxError):
        return value.strip('"\'')


def stable_fingerprint(stage: str, parent_exp_id: str | None, overrides: dict[str, Any]) -> str:
    payload = {
        "stage": stage,
        "parent_exp_id": parent_exp_id,
        "overrides": {key: repr(value) for key, value in sorted(overrides.items())},
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:6]


def shell_join(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def run_command(command: list[str], cwd: Path, log_path: Path) -> str:
    """
    subprocess stdout/stderr를 터미널에 실시간 출력하면서 같은 내용을 log 파일에도 저장한다.
    실패하면 CalledProcessError를 발생시켜 해당 EXP를 FAILED로 기록한다.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        header = f"$ {shell_join(command)}\n"
        print(header, end="")
        log_file.write(header)
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
            captured.append(line)

        return_code = process.wait()

    output = "".join(captured)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, output=output)
    return output


# ============================================================================
# Stage 정의
# ============================================================================


def preset_combinations(stage: str, custom_values: list[Any] | None = None) -> list[dict[str, Any]]:
    """Stage 하나에서 parent EXP 하나당 생성할 parameter override 조합을 반환한다."""

    if stage == "baseline":
        return [{}]

    if stage == "q2q3":
        q2_values = [64, 128, 256, 512]
        q3_values = [128, 256, 512, 1024]
        return [
            {"train.vae_c2_codebook_size": q2, "train.vae_c3_codebook_size": q3}
            for q2 in q2_values for q3 in q3_values
        ]

    if stage == "latent":
        values = custom_values or [64, 128, 256, 512]
        return [{"train.vae_embed_dim": value} for value in values]

    if stage == "quantizer_norm":
        return [
            {"train.vae_codebook_mode": mode, "train.vae_codebook_normalize": normalize}
            for mode in ["STE", "GUMBEL", "ROTATION"] for normalize in [False, True]
        ]

    if stage == "loss":
        cb_values = [0.25, 0.5, 1.0, 2.0]
        com_values = [0.1, 0.25, 1.0, 2.0]
        return [
            {"train.lambda_cb": cb, "train.lambda_com": com}
            for cb in cb_values for com in com_values
        ]

    if stage == "hidden":
        values = custom_values or [[512], [512, 256], [1024, 512]]
        return [{"train.vae_hidden_dims": value} for value in values]

    if stage == "lr":
        values = custom_values or [5e-5, 1e-4, 2e-4, 5e-4]
        return [{"train.learning_rate": value} for value in values]

    if stage == "simvq":
        values = custom_values or [False, True]
        return [{"train.vae_sim_vq": value} for value in values]

    if stage == "batch":
        values = custom_values or [128, 256, 512]
        return [{"train.batch_size": value} for value in values]

    if stage == "weight_decay":
        values = custom_values or [0.0, 0.01, 0.05]
        return [{"train.weight_decay": value} for value in values]

    if stage == "lambda_rec":
        values = custom_values or [0.5, 1.0, 2.0]
        return [{"train.lambda_rec": value} for value in values]

    if stage == "gumbel_t0":
        values = custom_values or [0.5, 1.0, 2.0]
        return [{"train.gumbel_t0": value} for value in values]

    if stage == "gumbel_min_t":
        values = custom_values or [0.05, 0.1, 0.2]
        return [{"train.gumbel_min_t": value} for value in values]

    if stage == "gumbel_anneal":
        values = custom_values or [2.9e-5, 5.8e-5, 1.16e-4]
        return [{"train.gumbel_anneal_rate": value} for value in values]

    if stage == "kmeans_n_init":
        values = custom_values or [5, 10, 20]
        return [{"train.kmeans_n_init": value} for value in values]

    if stage == "seed":
        # 탐색 과정의 parent가 이미 seed=42 결과이므로 기본값은 추가 검증 seed만 실행한다.
        values = custom_values or [123, 2026]
        return [{"train.seed": value} for value in values]

    raise ValueError(f"지원하지 않는 preset stage입니다: {stage}")


def custom_combinations(param: str, values: list[Any] | None) -> list[dict[str, Any]]:
    if not param:
        raise ValueError("custom Stage는 --param이 필요합니다. 예: --param train.epochs")
    if not values:
        raise ValueError("custom Stage는 --values가 필요합니다. 예: --values 400,800,1283")
    return [{param: value} for value in values]


# ============================================================================
# Parent EXP / 실행 계획 생성
# ============================================================================


def find_experiment_metadata(experiment_root: Path, exp_id: str) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for metadata_path in experiment_root.glob("stage_*/*/metadata.json"):
        metadata = load_json(metadata_path, {})
        if metadata.get("exp_id") == exp_id:
            matches.append((metadata_path.parent, metadata))

    if not matches:
        raise FileNotFoundError(
            f"parent EXP '{exp_id}'를 {experiment_root} 아래에서 찾지 못했습니다. "
            "이전 Stage 결과의 Exp ID를 그대로 --parents에 넣어주세요."
        )
    if len(matches) > 1:
        raise RuntimeError(f"같은 Exp ID가 여러 개 발견되었습니다: {exp_id}")
    return matches[0]


def resolve_parents(
    stage: str,
    parent_ids: list[str],
    experiment_root: Path,
    baseline_config: Path,
) -> list[tuple[str | None, Path]]:
    if stage in {"baseline", "q2q3"}:
        if parent_ids:
            raise ValueError(f"{stage} Stage에는 --parents를 사용하지 않습니다.")
        return [(None, baseline_config)]

    if not parent_ids:
        raise ValueError(
            f"{stage} Stage에는 이전 Stage에서 선택한 EXP를 --parents로 지정해야 합니다. "
            "예: --parents Q2Q3-003-xxxxxx,Q2Q3-008-yyyyyy"
        )

    parents: list[tuple[str | None, Path]] = []
    for exp_id in parent_ids:
        exp_dir, _ = find_experiment_metadata(experiment_root, exp_id)
        config_path = exp_dir / "config.gin"
        if not config_path.exists():
            raise FileNotFoundError(f"parent config가 없습니다: {config_path}")
        parents.append((exp_id, config_path))
    return parents


def stage_prefix(stage: str) -> str:
    mapping = {
        "baseline": "BASE",
        "q2q3": "Q2Q3",
        "latent": "LAT",
        "quantizer_norm": "QNT",
        "loss": "LOSS",
        "hidden": "HID",
        "lr": "LR",
        "simvq": "SIMVQ",
        "batch": "BATCH",
        "weight_decay": "WD",
        "lambda_rec": "LREC",
        "gumbel_t0": "GT0",
        "gumbel_min_t": "GMIN",
        "gumbel_anneal": "GANN",
        "kmeans_n_init": "KINIT",
        "seed": "SEED",
        "custom": "CUSTOM",
    }
    return mapping.get(stage, stage.upper()[:8])


def build_plans(
    stage: str,
    parents: list[tuple[str | None, Path]],
    combinations: list[dict[str, Any]],
    experiment_root: Path,
    global_overrides: dict[str, Any],
    limit: int | None,
) -> list[ExperimentPlan]:
    stage_dir = experiment_root / f"stage_{stage}"
    plans: list[ExperimentPlan] = []
    prefix = stage_prefix(stage)
    sequence = 1

    for parent_exp_id, base_config_path in parents:
        for combo in combinations:
            overrides = dict(combo)
            overrides.update(global_overrides)
            fingerprint = stable_fingerprint(stage, parent_exp_id, overrides)
            exp_id = f"{prefix}-{sequence:03d}-{fingerprint}"
            exp_dir = stage_dir / exp_id
            plans.append(
                ExperimentPlan(
                    exp_id=exp_id,
                    stage=stage,
                    parent_exp_id=parent_exp_id,
                    base_config_path=base_config_path,
                    overrides=overrides,
                    exp_dir=exp_dir,
                )
            )
            sequence += 1

    return plans[:limit] if limit is not None else plans


# ============================================================================
# 로그 파싱
# ============================================================================


def parse_float(text: str) -> float | None:
    try:
        value = float(text)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def last_regex_float(text: str, pattern: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    return parse_float(value)


def last_regex_int(text: str, pattern: str) -> int | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_training_history(train_log: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    train_rqvae.py가 출력하는 epoch/validation 로그를 구조화한다.

    이를 통해 total loss가 가장 낮은 epoch와 reconstruction loss가 가장 낮은 epoch를
    따로 확인할 수 있다. 두 loss가 다른 방향으로 움직이는 상황도 결과에서 바로 볼 수 있다.
    """

    train_pattern = re.compile(
        r"\[Epoch\s+(\d+)/(\d+)\]\s+loss=([0-9eE+\-.]+)\s*\|\s*rec=([0-9eE+\-.]+)\s*\|\s*"
        r"cb=([0-9eE+\-.]+)\s*\|\s*com=([0-9eE+\-.]+)\s*\|\s*rqvae=([0-9eE+\-.]+)\s*\|\s*"
        r"t=([0-9eE+\-.]+)\s*\|\s*global_step=(\d+)"
    )
    valid_pattern = re.compile(
        r"\[Validation epoch\s+(\d+)\]\s+loss=([0-9eE+\-.]+)\s*\|\s*rec=([0-9eE+\-.]+)\s*\|\s*"
        r"cb=([0-9eE+\-.]+)\s*\|\s*com=([0-9eE+\-.]+)\s*\|\s*rqvae=([0-9eE+\-.]+)"
    )

    rows: dict[int, dict[str, Any]] = {}
    for match in train_pattern.finditer(train_log):
        epoch = int(match.group(1))
        rows.setdefault(epoch, {})
        rows[epoch].update(
            {
                "epoch": epoch,
                "epochs_total": int(match.group(2)),
                "train_total_loss": float(match.group(3)),
                "train_rec_loss": float(match.group(4)),
                "train_codebook_loss": float(match.group(5)),
                "train_commitment_loss": float(match.group(6)),
                "train_rqvae_loss": float(match.group(7)),
                "gumbel_temperature": float(match.group(8)),
                "global_step": int(match.group(9)),
            }
        )

    for match in valid_pattern.finditer(train_log):
        epoch = int(match.group(1))
        rows.setdefault(epoch, {"epoch": epoch})
        rows[epoch].update(
            {
                "valid_total_loss": float(match.group(2)),
                "valid_rec_loss": float(match.group(3)),
                "valid_codebook_loss": float(match.group(4)),
                "valid_commitment_loss": float(match.group(5)),
                "valid_rqvae_loss": float(match.group(6)),
            }
        )

    history = [rows[epoch] for epoch in sorted(rows)]
    valid_rows = [row for row in history if row.get("valid_total_loss") is not None]
    summary: dict[str, Any] = {}

    if valid_rows:
        best_total = min(valid_rows, key=lambda row: row["valid_total_loss"])
        best_rec = min(valid_rows, key=lambda row: row["valid_rec_loss"])
        final_valid = valid_rows[-1]
        summary.update(
            {
                "best_valid_total_loss": best_total["valid_total_loss"],
                "best_valid_total_epoch": best_total["epoch"],
                "best_valid_rec_loss": best_rec["valid_rec_loss"],
                "best_valid_rec_epoch": best_rec["epoch"],
                "final_valid_total_loss": final_valid["valid_total_loss"],
                "final_valid_rec_loss": final_valid["valid_rec_loss"],
            }
        )

    return history, summary


def write_training_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_evaluation_metrics(eval_log: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "train_rec_loss": last_regex_float(eval_log, r"^Train Rec Loss\s*:\s*([0-9eE+\-.]+)"),
        "valid_rec_loss": last_regex_float(eval_log, r"^Valid Rec Loss\s*:\s*([0-9eE+\-.]+)"),
        "c2_event_consistency": last_regex_float(eval_log, r"^C2 Event Consistency\s*:\s*([0-9eE+\-.]+)"),
        "evaluated_multi_article_events": last_regex_int(eval_log, r"^Evaluated Multi-article Events\s*:\s*(\d+)"),
        "total_articles": last_regex_int(eval_log, r"^Total Articles\s*:\s*(\d+)"),
        "q2_used_codes": last_regex_int(eval_log, r"^Q2 Used Codes\s*:\s*(\d+)"),
        "q2_utilization": last_regex_float(eval_log, r"^Q2 Utilization\s*:\s*([0-9.]+)%"),
        "q3_used_codes": last_regex_int(eval_log, r"^Q3 Used Codes\s*:\s*(\d+)"),
        "q3_utilization": last_regex_float(eval_log, r"^Q3 Utilization\s*:\s*([0-9.]+)%"),
        "unique_c123": last_regex_int(eval_log, r"^Unique c123\s*:\s*(\d+)"),
        "collision_articles": last_regex_int(eval_log, r"^Collision Articles\s*:\s*(\d+)"),
        "collision_rate": last_regex_float(eval_log, r"^Collision Rate\s*:\s*([0-9.]+)%"),
        "delta_c2_similarity": last_regex_float(eval_log, r"^ΔC2 Semantic Similarity\s*:\s*([0-9eE+\-.]+)"),
        "delta_c3_similarity": last_regex_float(eval_log, r"^ΔC3 Semantic Similarity\s*:\s*([0-9eE+\-.]+)"),
        "c1_category_accuracy": last_regex_float(eval_log, r"^C1 Category Accuracy\s*:\s*([0-9eE+\-.]+)"),
    }

    # print_result()의 percentage 출력은 이미 0~100% 값이므로 CSV에서는 다시 0~1 비율로 맞춘다.
    for key in ["q2_utilization", "q3_utilization", "collision_rate"]:
        if metrics.get(key) is not None:
            metrics[key] = metrics[key] / 100.0

    return metrics


def parse_semantic_id_metrics(generate_log: str) -> dict[str, Any]:
    # Train / Validation / 전체의 통계가 순서대로 출력되므로 마지막 값을 사용하면 전체 unique article 통계다.
    return {
        "final_sid_uniqueness": last_regex_float(generate_log, r"^Final SID uniqueness\s*:\s*([0-9eE+\-.]+)"),
        "max_c4": last_regex_int(generate_log, r"^Max c4\s*:\s*(\d+)"),
    }


# ============================================================================
# 결과 파일 생성
# ============================================================================


def flatten_result(metadata: dict[str, Any], status: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "exp_id": metadata.get("exp_id"),
        "stage": metadata.get("stage"),
        "status": status.get("status"),
        "parent_exp_id": metadata.get("parent_exp_id"),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "duration_seconds": status.get("duration_seconds"),
        "error": status.get("error"),
    }

    for key, value in metadata.get("config", {}).items():
        row[key.replace("train.", "")] = value

    row.update(metrics)
    row["exp_dir"] = metadata.get("exp_dir")
    return row


def collect_stage_rows(stage_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not stage_dir.exists():
        return rows

    for exp_dir in sorted(path for path in stage_dir.iterdir() if path.is_dir()):
        metadata = load_json(exp_dir / "metadata.json", {})
        if not metadata:
            continue
        status = load_json(exp_dir / "status.json", {"status": "planned"})
        metrics = load_json(exp_dir / "metrics.json", {})
        rows.append(flatten_result(metadata, status, metrics))
    return rows


def ordered_fields(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "exp_id", "stage", "status", "parent_exp_id",
        "vae_embed_dim", "vae_c2_codebook_size", "vae_c3_codebook_size",
        "vae_codebook_mode", "vae_codebook_normalize", "vae_sim_vq",
        "lambda_rec", "lambda_cb", "lambda_com", "vae_hidden_dims",
        "learning_rate", "batch_size", "weight_decay", "epochs", "seed",
        *METRIC_COLUMNS,
        "duration_seconds", "started_at", "finished_at", "error", "exp_dir",
    ]
    all_keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)
    return [key for key in preferred if key in all_keys] + [key for key in all_keys if key not in preferred]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = ordered_fields(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def fmt(value: Any, percent: bool = False, digits: int = 4) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        if percent:
            return f"{value * 100:.2f}%"
        return f"{value:.{digits}f}"
    return str(value)


def write_stage_report(path: Path, stage: str, rows: list[dict[str, Any]]) -> None:
    """팀이 브라우저에서 바로 비교할 수 있는 간단한 Stage 결과 HTML."""

    completed = sum(row.get("status") == "completed" for row in rows)
    failed = sum(row.get("status") == "failed" for row in rows)
    planned = len(rows) - completed - failed

    table_rows = []
    for row in rows:
        mode = str(row.get("vae_codebook_mode") or "").replace("%QuantizeForwardMode.", "")
        table_rows.append(
            "<tr>"
            f"<td><b>{html.escape(str(row.get('exp_id', '')))}</b><br><small>{html.escape(str(row.get('parent_exp_id') or '-'))}</small></td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(row.get('vae_embed_dim', '-')))}</td>"
            f"<td>{html.escape(str(row.get('vae_c2_codebook_size', '-')))}</td>"
            f"<td>{html.escape(str(row.get('vae_c3_codebook_size', '-')))}</td>"
            f"<td>{html.escape(mode)}</td>"
            f"<td>{html.escape(str(row.get('vae_codebook_normalize', '-')))}</td>"
            f"<td>{fmt(row.get('lambda_cb'))}</td>"
            f"<td>{fmt(row.get('lambda_com'))}</td>"
            f"<td>{fmt(row.get('valid_rec_loss'), digits=6)}</td>"
            f"<td>{fmt(row.get('c2_event_consistency'), digits=4)}</td>"
            f"<td>{fmt(row.get('collision_rate'), percent=True)}</td>"
            f"<td>{fmt(row.get('q2_utilization'), percent=True)}</td>"
            f"<td>{fmt(row.get('q3_utilization'), percent=True)}</td>"
            f"<td>{fmt(row.get('delta_c2_similarity'), digits=4)}</td>"
            f"<td>{fmt(row.get('delta_c3_similarity'), digits=4)}</td>"
            f"<td>{fmt(row.get('final_sid_uniqueness'), percent=True)}</td>"
            f"<td>{fmt(row.get('max_c4'))}</td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RQ-VAE Stage Report · {html.escape(stage)}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans KR',sans-serif;margin:0;background:#f4f7fb;color:#172033}}
main{{max-width:1600px;margin:auto;padding:28px}} h1{{margin:0 0 8px;color:#173a67}} p{{color:#64748b}}
.summary{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}} .chip{{background:white;border:1px solid #dbe4ef;border-radius:14px;padding:12px 18px;box-shadow:0 4px 14px #1f3a5f0d}}
.note{{padding:15px 18px;background:#edf4ff;border:1px solid #c6d8ff;border-radius:14px;margin:18px 0}}
.table{{overflow:auto;background:white;border:1px solid #dbe4ef;border-radius:16px}} table{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}
th{{position:sticky;top:0;background:#173a67;color:white;text-align:left;padding:10px}} td{{padding:9px 10px;border-bottom:1px solid #edf1f6}} tr:hover td{{background:#f7faff}} small{{color:#8190a5}}
</style></head><body><main>
<h1>RQ-VAE Stage Report · {html.escape(stage)}</h1>
<p>자동 실행 결과 비교표입니다. 이 표는 자동 우승자를 확정하지 않고 다음 Stage의 parent 후보를 사람이 판단하기 위한 자료입니다.</p>
<div class="summary"><div class="chip">전체 <b>{len(rows)}</b></div><div class="chip">완료 <b>{completed}</b></div><div class="chip">실패 <b>{failed}</b></div><div class="chip">미실행/계획 <b>{planned}</b></div></div>
<div class="note"><b>해석 방향:</b> C2 Event Consistency ↑, Collision Rate ↓, ΔC2/ΔC3 ↑, Valid Rec Loss ↓. Q2/Q3 Utilization은 무조건 100%가 목표가 아니라 dead-code/과도한 codebook 크기를 확인하는 진단 지표입니다. Final SID uniqueness는 c4 적용 후 정상적으로 100%에 가까워야 합니다.</div>
<div class="table"><table><thead><tr>
<th>EXP / Parent</th><th>Status</th><th>Latent</th><th>Q2</th><th>Q3</th><th>Quantizer</th><th>Norm</th><th>λcb</th><th>λcom</th><th>Valid Rec↓</th><th>C2 Event↑</th><th>Collision↓</th><th>Q2 Util</th><th>Q3 Util</th><th>ΔC2↑</th><th>ΔC3↑</th><th>Final SID Unique</th><th>Max c4</th>
</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def rebuild_all_results(experiment_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for stage_dir in sorted(experiment_root.glob("stage_*")):
        if stage_dir.is_dir():
            rows.extend(collect_stage_rows(stage_dir))
    if rows:
        write_csv(experiment_root / "all_results.csv", rows)


# ============================================================================
# EXP 실행
# ============================================================================


def prepare_experiment(plan: ExperimentPlan, global_paths: dict[str, Path]) -> tuple[Path, dict[str, Any]]:
    plan.exp_dir.mkdir(parents=True, exist_ok=True)
    base_text = read_config(plan.base_config_path)

    # save_dir_root는 EXP마다 분리한다. 체크포인트가 서로 덮어써지는 것을 방지하기 위한 핵심 안전장치다.
    rqvae_dir = plan.exp_dir / "rqvae"
    semantic_dir = plan.exp_dir / "semantic_ids"
    overrides = dict(plan.overrides)
    overrides["train.save_dir_root"] = rqvae_dir.resolve().as_posix() + "/"

    config_text = set_config_values(base_text, overrides)
    config_path = plan.exp_dir / "config.gin"
    config_path.write_text(config_text, encoding="utf-8")

    metadata = {
        "exp_id": plan.exp_id,
        "stage": plan.stage,
        "parent_exp_id": plan.parent_exp_id,
        "created_at": now_iso(),
        "base_config_path": str(plan.base_config_path.resolve()),
        "config_path": str(config_path.resolve()),
        "exp_dir": str(plan.exp_dir.resolve()),
        "rqvae_dir": str(rqvae_dir.resolve()),
        "semantic_dir": str(semantic_dir.resolve()),
        "overrides": {key: gin_value(value) for key, value in plan.overrides.items()},
        "config": config_snapshot(config_text),
        "paths": {key: str(value.resolve()) for key, value in global_paths.items()},
    }
    json_dump(plan.exp_dir / "metadata.json", metadata)
    return config_path, metadata


def experiment_is_complete(exp_dir: Path) -> bool:
    status = load_json(exp_dir / "status.json", {})
    return (
        status.get("status") == "completed"
        and (exp_dir / "rqvae" / "checkpoint_final.pt").exists()
        and (exp_dir / "semantic_ids" / "article_semantic_ids.parquet").exists()
        and (exp_dir / "metrics.json").exists()
    )


def run_experiment(plan: ExperimentPlan, args: argparse.Namespace) -> None:
    if experiment_is_complete(plan.exp_dir) and not args.force:
        print(f"\n[SKIP] {plan.exp_id}: 이미 완료된 EXP입니다. --force가 없으므로 건너뜁니다.")
        return

    global_paths = {
        "rqvae_root": args.rqvae_root,
        "experiment_root": args.experiment_root,
        "baseline_config": args.baseline_config,
    }
    config_path, metadata = prepare_experiment(plan, global_paths)

    if args.dry_run:
        json_dump(
            plan.exp_dir / "status.json",
            {"status": "planned", "updated_at": now_iso(), "dry_run": True},
        )
        print(f"[PLAN] {plan.exp_id} | parent={plan.parent_exp_id or '-'} | overrides={plan.overrides}")
        return

    config_text = read_config(config_path)
    dataset_folder = config_string(config_text, "train.dataset_folder")
    data_dir = Path(dataset_folder)
    if not data_dir.is_absolute():
        data_dir = args.rqvae_root / data_dir

    q2_size = config_int(config_text, "train.vae_c2_codebook_size")
    q3_size = config_int(config_text, "train.vae_c3_codebook_size")
    checkpoint = plan.exp_dir / "rqvae" / "checkpoint_final.pt"
    semantic_dir = plan.exp_dir / "semantic_ids"
    sid_path = semantic_dir / "article_semantic_ids.parquet"

    train_cmd = [sys.executable, "train_rqvae.py", str(config_path.resolve())]
    generate_cmd = [
        sys.executable,
        "generate_semantic_ids.py",
        "--data_dir", str(data_dir.resolve()),
        "--checkpoint", str(checkpoint.resolve()),
        "--output_dir", str(semantic_dir.resolve()),
        "--batch_size", str(args.sid_batch_size),
        "--num_workers", str(args.sid_num_workers),
    ]
    evaluate_cmd = [
        sys.executable,
        "evaluate/evaluate_all.py",
        "--sid", str(sid_path.resolve()),
        "--data-dir", str(data_dir.resolve()),
        "--checkpoint", str(checkpoint.resolve()),
        "--split", args.eval_split,
        "--q2-size", str(q2_size),
        "--q3-size", str(q3_size),
        "--pairs", str(args.pairs),
    ]

    start = time.time()
    status = {
        "status": "running",
        "started_at": now_iso(),
        "finished_at": None,
        "duration_seconds": None,
        "error": None,
    }
    json_dump(plan.exp_dir / "status.json", status)

    print("\n" + "=" * 100)
    print(f"START {plan.exp_id} | stage={plan.stage} | parent={plan.parent_exp_id or '-'}")
    print(f"Overrides: {plan.overrides}")
    print("=" * 100)

    try:
        train_output = run_command(train_cmd, args.rqvae_root, plan.exp_dir / "train.log")
        if not checkpoint.exists():
            raise FileNotFoundError(f"학습은 종료됐지만 checkpoint_final.pt가 없습니다: {checkpoint}")

        semantic_output = run_command(generate_cmd, args.rqvae_root, plan.exp_dir / "semantic_ids.log")
        if not sid_path.exists():
            raise FileNotFoundError(f"Semantic ID 생성 후 article_semantic_ids.parquet가 없습니다: {sid_path}")

        eval_output = run_command(evaluate_cmd, args.rqvae_root, plan.exp_dir / "evaluate.log")

        history, history_summary = parse_training_history(train_output)
        write_training_history(plan.exp_dir / "training_history.csv", history)

        metrics = parse_evaluation_metrics(eval_output)
        metrics.update(parse_semantic_id_metrics(semantic_output))
        metrics.update(history_summary)
        json_dump(plan.exp_dir / "metrics.json", metrics)

        status.update(
            {
                "status": "completed",
                "finished_at": now_iso(),
                "duration_seconds": round(time.time() - start, 3),
            }
        )
        json_dump(plan.exp_dir / "status.json", status)
        print(f"\n[DONE] {plan.exp_id} ({status['duration_seconds']:.1f}s)")

    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "finished_at": now_iso(),
                "duration_seconds": round(time.time() - start, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        json_dump(plan.exp_dir / "status.json", status)
        print(f"\n[FAILED] {plan.exp_id}: {status['error']}", file=sys.stderr)
        if args.fail_fast:
            raise



def preflight_check(args: argparse.Namespace) -> None:
    """실제 학습을 시작하기 전에 경로/필수 입력 파일을 한 번만 확인한다."""

    required_scripts = [
        args.rqvae_root / "train_rqvae.py",
        args.rqvae_root / "generate_semantic_ids.py",
        args.rqvae_root / "evaluate" / "evaluate_all.py",
    ]
    missing_scripts = [path for path in required_scripts if not path.exists()]
    if missing_scripts:
        raise FileNotFoundError("필수 실행 파일이 없습니다: " + ", ".join(map(str, missing_scripts)))

    baseline_text = read_config(args.baseline_config)
    dataset_folder = config_string(baseline_text, "train.dataset_folder")
    data_dir = Path(dataset_folder)
    if not data_dir.is_absolute():
        data_dir = args.rqvae_root / data_dir

    required_data = [
        data_dir / "article_master.parquet",
        data_dir / "validation_article_master.parquet",
        data_dir / "article_embeddings.npy",
    ]
    missing_data = [path for path in required_data if not path.exists()]
    if missing_data:
        raise FileNotFoundError(
            "RQ-VAE 실험에 필요한 EB-NeRD/MIND 입력 파일이 아직 준비되지 않았습니다. "
            "누락: " + ", ".join(map(str, missing_data))
        )


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RQ-VAE Stage별 hyperparameter 실험 자동 실행기",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "baseline", "q2q3", "latent", "quantizer_norm", "loss", "hidden", "lr",
            "simvq", "batch", "weight_decay", "lambda_rec", "gumbel_t0", "gumbel_min_t",
            "gumbel_anneal", "kmeans_n_init", "seed", "custom",
        ],
        help="실행할 Stage. 이전 Stage의 TOP EXP는 --parents로 전달합니다.",
    )
    parser.add_argument("--dataset", default="ebnerd", choices=sorted(DATASET_CONFIGS))
    parser.add_argument(
        "--parents",
        default="",
        help="이전 Stage에서 선택한 Exp ID 목록. comma로 구분. q2q3/baseline에는 사용하지 않음.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
        help="실험 결과 루트. 기본: RQVAE/out/experiments/<dataset>",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Baseline gin 경로를 직접 지정할 때 사용. 기본은 dataset별 configs/rqvae_*.gin",
    )

    # 공통 학습 budget override. Stage 내 모든 EXP에 같은 값을 적용해야 공정한 비교가 된다.
    parser.add_argument("--epochs", type=int, default=None, help="모든 EXP의 epochs를 같은 값으로 강제할 때 사용")
    parser.add_argument("--eval-every-epochs", type=int, default=None, help="Validation 평가 주기를 동일하게 강제")

    # Stage preset 값을 바꾸거나 custom 파라미터를 실험할 때 사용한다.
    parser.add_argument(
        "--values",
        default=None,
        help="단일 parameter Stage 후보를 덮어쓸 값. 일반 값은 comma, list 값은 semicolon 사용. 예: 64,128,256 또는 '[512];[512,256]'",
    )
    parser.add_argument("--param", default=None, help="custom Stage의 gin key. 예: train.epochs")

    # 평가 / Semantic ID 생성 실행값. 모델 하이퍼파라미터가 아니라 실행 비용 관련 값이다.
    parser.add_argument("--pairs", type=int, default=50000, help="Semantic similarity pair sampling 수")
    parser.add_argument("--eval-split", default="all", choices=["all", "train", "validation"])
    parser.add_argument("--sid-batch-size", type=int, default=512)
    parser.add_argument("--sid-num-workers", type=int, default=0)

    # 운영 안전 옵션.
    parser.add_argument("--dry-run", action="store_true", help="config/계획만 만들고 학습은 실행하지 않음")
    parser.add_argument("--limit", type=int, default=None, help="앞 N개 EXP만 실행. smoke test에 유용")
    parser.add_argument("--force", action="store_true", help="완료된 EXP도 다시 실행")
    parser.add_argument("--fail-fast", action="store_true", help="한 EXP가 실패하면 Stage 전체를 즉시 중단")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs는 1 이상이어야 합니다.")
    if args.eval_every_epochs is not None and args.eval_every_epochs <= 0:
        raise ValueError("--eval-every-epochs는 1 이상이어야 합니다.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit는 1 이상이어야 합니다.")
    if args.pairs <= 0:
        raise ValueError("--pairs는 1 이상이어야 합니다.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    args.rqvae_root = SCRIPT_DIR
    baseline_config = args.config or DATASET_CONFIGS[args.dataset]
    if not baseline_config.is_absolute():
        baseline_config = args.rqvae_root / baseline_config
    args.baseline_config = baseline_config.resolve()

    experiment_root = args.experiment_root or (args.rqvae_root / "out" / "experiments" / args.dataset)
    if not experiment_root.is_absolute():
        experiment_root = args.rqvae_root / experiment_root
    args.experiment_root = experiment_root.resolve()
    args.experiment_root.mkdir(parents=True, exist_ok=True)

    parent_ids = [item.strip() for item in args.parents.split(",") if item.strip()]
    parents = resolve_parents(args.stage, parent_ids, args.experiment_root, args.baseline_config)
    custom_values = parse_values_csv(args.values)

    if args.stage == "custom":
        combinations = custom_combinations(args.param, custom_values)
    else:
        combinations = preset_combinations(args.stage, custom_values)

    # --epochs / --eval-every-epochs는 검색 parameter가 아니라 동일 Stage의 공통 budget override다.
    global_overrides: dict[str, Any] = {}
    if args.epochs is not None:
        global_overrides["train.epochs"] = args.epochs
    if args.eval_every_epochs is not None:
        global_overrides["train.eval_every_epochs"] = args.eval_every_epochs

    plans = build_plans(
        stage=args.stage,
        parents=parents,
        combinations=combinations,
        experiment_root=args.experiment_root,
        global_overrides=global_overrides,
        limit=args.limit,
    )

    # dry-run은 데이터가 없어도 실험 계획을 검토할 수 있어야 하므로 실제 실행일 때만 검사한다.
    if not args.dry_run:
        preflight_check(args)

    print("\n" + "=" * 100)
    print("RQ-VAE STAGED EXPERIMENT RUNNER")
    print("=" * 100)
    print(f"Dataset         : {args.dataset}")
    print(f"Stage           : {args.stage}")
    print(f"Baseline config : {args.baseline_config}")
    print(f"Experiment root : {args.experiment_root}")
    print(f"Parents         : {', '.join(parent_ids) if parent_ids else '-'}")
    print(f"Planned EXPs    : {len(plans)}")
    print(f"Dry run         : {args.dry_run}")
    print("=" * 100)

    for index, plan in enumerate(plans, start=1):
        print(f"\n[{index}/{len(plans)}] {plan.exp_id}")
        run_experiment(plan, args)

        # EXP 하나가 끝날 때마다 CSV/HTML을 갱신한다. 중간에 서버가 종료되어도 현재까지 결과가 남는다.
        stage_dir = args.experiment_root / f"stage_{args.stage}"
        rows = collect_stage_rows(stage_dir)
        write_csv(stage_dir / "stage_results.csv", rows)
        write_stage_report(stage_dir / "stage_report.html", args.stage, rows)
        rebuild_all_results(args.experiment_root)

    stage_dir = args.experiment_root / f"stage_{args.stage}"
    rows = collect_stage_rows(stage_dir)
    completed = sum(row.get("status") == "completed" for row in rows)
    failed = sum(row.get("status") == "failed" for row in rows)

    print("\n" + "=" * 100)
    print("STAGE FINISHED")
    print("=" * 100)
    print(f"Completed : {completed}")
    print(f"Failed    : {failed}")
    print(f"CSV       : {stage_dir / 'stage_results.csv'}")
    print(f"HTML      : {stage_dir / 'stage_report.html'}")
    print(f"All CSV   : {args.experiment_root / 'all_results.csv'}")
    print("다음 단계: stage_results.csv를 검토한 뒤 선택한 EXP ID를 다음 Stage의 --parents에 넣으세요.")
    print("=" * 100)


if __name__ == "__main__":
    main()
