"""NRMS 'e5' 모드에 넣을 recon 표를 만든다 (교수님 제안: NRMS 뉴스 입력 = RQ-VAE 복원 벡터).

recon[row] = decoder(Q1[c1] + Q2[c2] + Q3[c3])   (UNI RQ-VAE best_rec, 저장된 SID 정수 사용)

- SID는 nrms_original_ebnerd_v2/tables/sid_codes.npy (= NRMS SID 모드가 쓴 코드 그대로)
- 기사 행은 tables/article_id2row.pkl 의 값 (train + validation 전체 12,860개)
- 표 모양은 원본 article_embeddings.npy와 같게 두고, 기사가 없는 행은 0
- 원본 파일은 읽기만 한다. --out 이 이미 있으면 멈춘다.
"""
import argparse, hashlib, inspect, pickle, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "RQVAE"))
UNI_SHA256 = "dcb276f8bc87237f272668df4b51a0aa086f7d3817115e5c072ad5c032be73ba"  # WD-001-UNI-lu0.05-m0.5 best_rec (epoch 180)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="UNI RQ-VAE checkpoint_best_rec.pt")
    ap.add_argument("--tables_dir", required=True, help="nrms_original_ebnerd_v2 tables 폴더 (sid_codes.npy, article_id2row.pkl)")
    ap.add_argument("--e5_path", required=True, help="원본 article_embeddings.npy (모양 · 자기복원 cosine 확인용)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 가 이미 있습니다 (덮어쓰지 않음)")

    digest = sha256(a.ckpt)
    print(f"[recon] checkpoint sha256 {digest[:12]}…  ({'UNI 일치' if digest == UNI_SHA256 else '주의: UNI checkpoint와 다름'})")
    if digest != UNI_SHA256:
        raise SystemExit("UNI checkpoint가 아닙니다. --ckpt 경로를 확인하세요.")

    from modules.rqvae import RqVae
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["model_config"])
    accepted = set(inspect.signature(RqVae.__init__).parameters) - {"self"}
    model = RqVae(**{k: v for k, v in cfg.items() if k in accepted})
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    print(f"[recon] epoch {ck['epoch'] + 1} · c1 {cfg['num_categories']} · c2 {cfg['c2_codebook_size']} · c3 {cfg['c3_codebook_size']}")

    T = Path(a.tables_dir)
    codes = np.load(T / "sid_codes.npy")
    id2row = pickle.load(open(T / "article_id2row.pkl", "rb"))
    rows = np.array(sorted(set(int(r) for r in id2row.values())), dtype=np.int64)
    E5 = np.load(a.e5_path, mmap_mode="r")
    assert rows.max() < len(codes) <= E5.shape[0], (rows.max(), len(codes), E5.shape)
    print(f"[recon] 기사 {len(rows):,}개 · 표 {E5.shape}")

    c = torch.from_numpy(codes[rows].astype(np.int64))
    with torch.inference_mode():
        z_q = (model.quantizer_1.get_item_embeddings(c[:, 0]) + model.quantizer_2.get_item_embeddings(c[:, 1])
               + model.quantizer_3.get_item_embeddings(c[:, 2]))
        xh = model.decode(z_q).numpy().astype(np.float32)

    table = np.zeros(E5.shape, dtype=np.float32)
    table[rows] = xh

    x = np.asarray(E5[rows], dtype=np.float32)
    cos = (x * xh).sum(1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(xh, axis=1), 1e-12)
    print(f"[recon] 자기복원 cosine cos(E5, recon) 평균 {cos.mean():.4f} (정량평가 3의 UNI 값 0.9275와 같아야 함)")
    uniq = len({tuple(r) for r in codes[rows].tolist()})
    print(f"[recon] 서로 다른 (c1,c2,c3) {uniq:,}개 → 서로 다른 recon 벡터 {uniq:,}개 (같은 SID면 같은 벡터)")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, table)
    print(f"[recon] 저장 {out}")


if __name__ == "__main__":
    main()
