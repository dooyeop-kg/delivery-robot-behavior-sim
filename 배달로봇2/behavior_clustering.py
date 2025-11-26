import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1) 데이터 불러오기
# ==========================================
df = pd.read_csv("combined_features.csv")
print("데이터 불러오기 완료!")
print(df.head())

# ==========================================
# 2) 클러스터링에 사용할 피처 선택
# ==========================================
# DE가 제공한 features 파일 기준으로 아래 컬럼들 교체 가능
# 예: speed, direction_change_rate, stop_rate, straightness 등
# 실제 들어있는 컬럼명에 맞게 아래 목록 수정 가능
feature_cols = [
    'path_length', 
    'displacement', 
    'straightness',
    'traj_avg_speed',
    'stop_rate',
    'direction_change_rate'
]

# 혹시 컬럼이 없는 경우 자동 필터링
feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols]

print("\n사용된 feature 목록:", feature_cols)

# ==========================================
# 3) 스케일링
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 4) K-means 클러스터링
# ==========================================
k = 4   # 원하는 행동 유형 개수
kmeans = KMeans(n_clusters=k, random_state=42)
df['behavior_type'] = kmeans.fit_predict(X_scaled)

print("\n클러스터링 완료!")
print(df['behavior_type'].value_counts())

# ==========================================
# 5) 장소별 행동 유형 비율 계산
# ==========================================
loc_ratio = df.groupby(['location', 'behavior_type']).size().unstack(fill_value=0)
loc_ratio = loc_ratio.div(loc_ratio.sum(axis=1), axis=0)

print("\n장소별 행동 타입 비율:")
print(loc_ratio)

loc_ratio.to_csv("location_behavior_ratio_v2.csv")
df.to_csv("combined_features_with_behavior.csv", index=False)

print("\n파일 저장 완료!")
print("- location_behavior_ratio_v2.csv")
print("- combined_features_with_behavior.csv")

# ==========================================
# 6) 행동 유형 시각화 (옵션)
# ==========================================
plt.figure(figsize=(8,5))
df['behavior_type'].value_counts().sort_index().plot(kind='bar')
plt.title("Behavior Type Distribution")
plt.xlabel("Behavior Type")
plt.ylabel("Count")
plt.savefig("behavior_type_distribution_v2.png", dpi=300)
plt.close()

print("\n그래프 저장 완료! (behavior_type_distribution_v2.png)")
