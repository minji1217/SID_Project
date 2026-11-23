"""Transformer 에 넣을 1pos4neg 파일 확인: 행 수, 후보 5개·정답 1개, 정답이 첫 자리인 비율.

    python experiments/v2_yeomin_rerun/check_candidates.py <train> <validation_half> <test>

정답이 첫 자리인 비율이 0.5 를 넘으면 후보 순서를 안 섞은 파일이다 (argmax 동점이 정답으로 처리되어 Top-1 이 부풂). 그때 멈춘다.
"""
import sys

import pandas as pd

bad = False
for path in sys.argv[1:]:
    d = pd.read_parquet(path, columns=["candidate_labels"])
    labels = d.candidate_labels.map(list)
    n5 = bool((labels.map(len) == 5).all())
    p1 = bool((labels.map(sum) == 1).all())
    first = float(labels.map(lambda x: x.index(1) == 0 if 1 in x else False).mean())
    print(f"{path.split('/')[-1]:<45} 행 {len(d):>9,} · 후보 5개 {n5} · 정답 1개 {p1} · 정답 첫 자리 비율 {first:.3f}")
    bad |= (not n5) or (not p1) or first > 0.5
if bad:
    sys.exit("문제가 있는 파일이 있습니다 (후보 수·정답 수·섞기 확인)")
