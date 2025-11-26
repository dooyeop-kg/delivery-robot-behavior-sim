import pandas as pd
import glob

# ==========================================
# 1) features_A~E.csv 자동 탐지 & 로드
# ==========================================
file_list = glob.glob("features_*.csv")

print("불러온 features 파일 목록:")
print(file_list)

if len(file_list) == 0:
    print(" features_A~E 파일을 찾을 수 없습니다.")
    exit()

# ==========================================
# 2) 모든 장소(A~E) feature 데이터 불러오기
# ==========================================
df_list = []
for file in file_list:
    temp = pd.read_csv(file)
    
    # 파일명에서 A, B, C, D, E 추출해서 location 컬럼 추가
    location = file.split("_")[1].replace(".csv", "")
    temp["location"] = location
    
    df_list.append(temp)

# ==========================================
# 3) 하나의 DataFrame으로 합치기
# ==========================================
df = pd.concat(df_list, ignore_index=True)

print("\n전체 feature 데이터 합치기 완료!")
print(df.head())

# 저장
df.to_csv("combined_features.csv", index=False)
print("\ncombined_features.csv 저장 완료!")
