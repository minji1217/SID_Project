"""기존 UNI(A) checkpoint 안의 gin_config 와 이번 gin 의 하이퍼파라미터가 같은지 비교한다.

dataset_folder / save_dir_root / pretrained_rqvae_path 는 경로라 비교에서 뺀다.
그 밖에 하나라도 다르면 종료 코드 1 로 멈춘다.

사용: python check_against_a.py <A checkpoint_best_rec.pt> <이번 rqvae_uni.gin> <학습 코드 RQVAE 폴더>
"""
import re
import sys
from pathlib import Path

ckpt, gin_path, rqvae_dir = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, rqvae_dir)  # checkpoint 안의 modules.* 객체를 풀기 위해

import torch  # noqa: E402

ck = torch.load(ckpt, map_location="cpu", weights_only=False)
a_gin = ck.get("gin_config")
if not a_gin:
    raise SystemExit("A checkpoint 에 gin_config 가 없습니다.")

SKIP = {"train.dataset_folder", "train.save_dir_root", "train.pretrained_rqvae_path"}


def parse(text):
    vals = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(train\.[A-Za-z0-9_]+)\s*=\s*(.+?)\s*(#.*)?$", line)
        if m:
            vals[m.group(1)] = m.group(2).strip()
    return vals


a, b = parse(a_gin), parse(gin_path.read_text(encoding="utf-8"))
def norm(v):
    v = str(v).replace("'", '"').replace(" ", "")
    if v.startswith("%"):          # %modules.quantize.QuantizeForwardMode.STE == %QuantizeForwardMode.STE
        return "%" + ".".join(v[1:].split(".")[-2:])
    try:
        return repr(float(v))      # 5.8e-05 == 0.000058
    except ValueError:
        return v


diff = [(k, a.get(k), b.get(k)) for k in sorted(set(a) | set(b))
        if k not in SKIP and norm(str(a.get(k))) != norm(str(b.get(k)))]
print(f"[A 비교] A epoch {ck.get('epoch', '?')} · 비교 항목 {len(set(a) | set(b)) - len(SKIP & (set(a) | set(b)))}개")
for k in ("train.lambda_uniq", "train.uniqueness_margin", "train.vae_embed_dim", "train.vae_c3_codebook_size", "train.seed"):
    print(f"   {k}: A={a.get(k)} · 이번={b.get(k)}")
if diff:
    for k, x, y in diff:
        print(f"   다름 {k}: A={x} · 이번={y}")
    raise SystemExit(1)
print("   A 와 하이퍼파라미터가 모두 같습니다 (경로 제외).")
