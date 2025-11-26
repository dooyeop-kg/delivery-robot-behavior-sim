import pandas as pd

ab = pd.read_csv("abnormal_features.csv")
print(" 이상 데이터 컬럼 목록:")
print(list(ab.columns))

print("\n 이상 데이터 샘플 5개:")
print(ab.head())
