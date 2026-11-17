"""UNI(WD-001-UNI-lu0.05-m0.5) 학습 설정으로 gin 파일을 만든다.

원본: uniqueness-loss-experiment 브랜치 RQVAE/configs/rqvae_ebnerd_wd001_uni.gin
바꾸는 것만 덮어쓴다.
- dataset_folder  : 새 전처리 출력 (exports/rqvae_inputs, 절대 경로)
- lambda_uniq     : 0.05 (A checkpoint 의 gin_config 값. 원본 gin 파일에는 0.1 로 적혀 있음)
- save_dir_root   : 새 결과 폴더
- seed            : 인자로 받음 (기본 42 = A 와 같음)
나머지 하이퍼파라미터(구조, loss 가중치, lr, early stopping 등)는 원본 그대로다.

사용:
  python make_gin.py --src <원본 gin> --out <새 gin> --dataset_folder <폴더> --save_dir <폴더> --seed 42
"""
import argparse
import re
from pathlib import Path

OVERRIDE_KEYS = ("train.dataset_folder", "train.lambda_uniq", "train.save_dir_root", "train.seed")

# A 와 같아야 하는 값. 원본 gin 이 바뀌었으면 멈춘다.
EXPECTED = {
    "train.vae_embed_dim": "128",
    "train.vae_hidden_dims": "[512]",
    "train.vae_num_categories": "25",
    "train.vae_c2_codebook_size": "128",
    "train.vae_c3_codebook_size": "512",
    "train.learning_rate": "0.0002",
    "train.weight_decay": "0.0",
    "train.lambda_cb": "0.25",
    "train.lambda_com": "0.1",
    "train.uniqueness_margin": "0.5",
    "train.early_stopping": "True",
    "train.early_stopping_min_epochs": "200",
    "train.early_stopping_patience": "20",
    "train.early_stopping_min_delta": "0.001",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset_folder", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_uniq", type=float, default=0.05)
    ap.add_argument("--extra", nargs="*", default=[], help="추가 override (스모크 테스트용) 예: train.epochs=1")
    a = ap.parse_args()

    lines = Path(a.src).read_text(encoding="utf-8").splitlines()
    # Colab 에서 저장할 때 들어간 첫 줄 '%%writefile ...' 은 gin 문법이 아니므로 뺀다
    lines = [l for l in lines if not l.startswith("%%")]

    values = {}
    for l in lines:
        m = re.match(r"^\s*(train\.[A-Za-z0-9_]+)\s*=\s*(.+?)\s*$", l)
        if m:
            values[m.group(1)] = m.group(2)
    for k, v in EXPECTED.items():
        if values.get(k) != v:
            raise SystemExit(f"원본 gin 의 {k} = {values.get(k)} (A 기대값 {v}). 원본 설정이 바뀌었는지 확인하세요.")

    overrides = {
        "train.dataset_folder": f'"{Path(a.dataset_folder).resolve()}"',
        "train.lambda_uniq": repr(a.lambda_uniq),
        "train.save_dir_root": f'"{Path(a.save_dir).resolve()}"',
        "train.seed": str(a.seed),
    }
    for e in a.extra:
        k, v = e.split("=", 1)
        overrides[k.strip()] = v.strip()

    out, seen = [], set()
    for l in lines:
        m = re.match(r"^\s*(train\.[A-Za-z0-9_]+)\s*=", l)
        if m and m.group(1) in overrides:
            k = m.group(1)
            out.append(f"{k} = {overrides[k]}   # override (rqvae_uni_fixedcluster_v1)")
            seen.add(k)
        else:
            out.append(l)
    missing = [k for k in overrides if k not in seen]
    if missing:
        out.append("")
        out += [f"{k} = {overrides[k]}   # override (rqvae_uni_fixedcluster_v1)" for k in missing]

    Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[gin] {a.out}")
    for k in OVERRIDE_KEYS:
        print(f"      {k} = {overrides[k]}")
    for e in a.extra:
        print(f"      (extra) {e}")


if __name__ == "__main__":
    main()
