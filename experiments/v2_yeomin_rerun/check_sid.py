"""② SID 생성 결과 확인. h/, r/ 중 있는 것을 요약하고, 둘 다 있으면 붙은 사건이 같은지 본다.

    python experiments/v2_yeomin_rerun/check_sid.py <V2>/sid

확인하는 것
  - 기사 수 (Train 10,419 + Validation-only 2,741 = 13,160), (c1,c2,c3,c4) 중복 0, c1~c4 최댓값
  - 매칭 방법별 건수, 방법별 match_cosine 분포 (코사인은 기준값 없이 가장 높은 사건에 붙으므로 값이 몰리면 배정이 무작위에 가까움)
  - H 와 R 은 따로 학습한 모델이라 h 가 달라서, 붙는 사건도 다를 수 있다 (비율만 출력)
  - _check_r1code_with_h_ckpt/ 가 있으면: R1 코드를 H checkpoint 로 돌린 결과가 H 와 같은 사건에 100% 붙어야 한다
    (코사인을 둘 다 h 로 하는지 코드 확인. 다르면 멈춤)
"""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
maps = {}
for v in ["h", "r"]:
    d = root / v
    if not (d / "article_semantic_ids.parquet").exists():
        continue
    a = pd.read_parquet(d / "article_semantic_ids.parquet")
    m = pd.read_parquet(d / "validation_article_train_event_mapping.parquet")
    maps[v] = m
    dup = int(a.duplicated(["c1", "c2", "c3", "c4"]).sum())
    print(f"[{v}] 기사 {len(a):,} (train {int((a.split == 'train').sum()):,} · validation {int((a.split == 'validation').sum()):,}) "
          f"· SID 중복 {dup} · 최댓값 c1..c4 {a[['c1', 'c2', 'c3', 'c4']].max().tolist()}")
    if len(a) != 13160 or dup:
        print(f"[{v}] 주의: 기사 수가 13,160 이 아니거나 SID 중복이 있습니다")
    print(f"[{v}] 매칭 방법: " + " · ".join(f"{k} {n:,}" for k, n in m.match_method.value_counts().items()))
    print(f"[{v}] 개체 있는 validation-only 기사: {(m.match_article_entity_count > 0).sum():,} / {len(m):,}")
    q = m.groupby("match_method").match_cosine.quantile([0.0, 0.25, 0.5, 0.75]).unstack().round(3)
    q.columns = ["min", "25%", "50%", "75%"]
    print(f"[{v}] match_cosine 분포\n{q.to_string()}")
    t = pd.read_parquet(d / "train_event_c2_mapping.parquet")
    top = t.event_c2.value_counts(normalize=True)
    used = m.event_c2.nunique() if "event_c2" in m else float("nan")
    print(f"[{v}] Train 사건 C2 ({t.source.iloc[0]}): 사건 {len(t):,}개가 받은 C2 종류 {t.event_c2.nunique()} / 128 · "
          f"가장 많은 C2 {top.index[0]}번이 {top.iloc[0]:.1%} · validation-only 가 실제로 물려받은 C2 종류 {used}")
    if top.iloc[0] > 0.5:
        print(f"[{v}] 주의: 사건 절반 이상이 같은 C2 하나를 받았습니다 (사건 대표 벡터와 Q2 공간이 안 맞을 수 있음)")

def compare(a, b, label, strict):
    x = a.merge(b, on="article_id", suffixes=("_a", "_b"))
    same = float((x.matched_train_event_id_a == x.matched_train_event_id_b).mean())
    c2 = float((x.event_c2_a == x.event_c2_b).mean()) if "event_c2_a" in x else float("nan")
    print(f"[{label}] 같은 사건에 붙은 비율 {same:.4f} · 물려받은 C2 가 같은 비율 {c2:.3f}")
    if strict and same != 1.0:
        sys.exit(f"[{label}] 1.0 이어야 합니다. R1 코드의 코사인이 h 로 되어 있는지 확인하세요.")


if {"h", "r"} <= maps.keys():
    compare(maps["h"], maps["r"], "H vs R (모델이 달라 다를 수 있음)", strict=False)

chk = root / "_check_r1code_with_h_ckpt" / "validation_article_train_event_mapping.parquet"
if "h" in maps and chk.exists():
    compare(maps["h"], pd.read_parquet(chk), "코드 확인: R1 코드 + H checkpoint vs H (1.0 이어야 함)", strict=True)
