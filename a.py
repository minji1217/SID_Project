import pandas as pd

# Parquet 파일 읽기
df = pd.read_parquet('data/train/history.parquet')

# 엑셀(XLSX) 파일로 저장
df.to_excel('data/train/history.xlsx', index=False)