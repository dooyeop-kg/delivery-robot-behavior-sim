import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1) 데이터 로드
# ============================================
file_pattern = "clean_*_10000_people.csv"
file_list = glob.glob(file_pattern)

print("\n불러온 파일 목록:", file_list)

if len(file_list) == 0:
    print("⚠ CSV 파일이 없습니다! clean_*.csv 파일을 넣어주세요.")
    exit()

dfs = [pd.read_csv(f) for f in file_list]
df = pd.concat(dfs, ignore_index=True)

print("데이터 로드 완료:", df.shape)


# ============================================
# 2) 파생 피처 생성 함수
# ============================================
def compute_path_length(row):
    dist = 0
    for i in range(9):
        x1, y1 = row[f'x_{i}'], row[f'y_{i}']
        x2, y2 = row[f'x_{i+1}'], row[f'y_{i+1}']
        dist += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def compute_displacement(row):
    return np.sqrt((row['x_9'] - row['x_0'])**2 + (row['y_9'] - row['y_0'])**2)

def compute_straightness(row):
    return row['displacement'] / row['path_length'] if row['path_length'] != 0 else 0

def compute_traj_avg_speed(row):
    return row['path_length'] / 9

def compute_stop_rate(row):
    return row['stop_count'] / 9

def compute_direction_change_rate(row):
    return row['direction_change'] / 9


# ============================================
# 3) 파생 피처 생성
# ============================================
df['path_length'] = df.apply(compute_path_length, axis=1)
df['displacement'] = df.apply(compute_displacement, axis=1)
df['straightness'] = df.apply(compute_straightness, axis=1)
df['traj_avg_speed'] = df.apply(compute_traj_avg_speed, axis=1)
df['stop_rate'] = df.apply(compute_stop_rate, axis=1)
df['direction_change_rate'] = df.apply(compute_direction_change_rate, axis=1)

print("파생 피처 생성 완료")


# ============================================
# 4) KMeans 보행자 행동 유형 클러스터링
# ============================================
features = [
    'path_length', 'displacement', 'straightness',
    'traj_avg_speed', 'stop_rate', 'direction_change_rate'
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df['behavior_type'] = kmeans.fit_predict(X_scaled)

print("\nbehavior_type 분포:")
print(df['behavior_type'].value_counts())


# ============================================
# 5) 클러스터별 프로파일 분석
# ============================================
cluster_profile = df.groupby('behavior_type')[features].mean()
cluster_profile.to_csv("behavior_type_feature_profile.csv")

print("\n 클러스터별 행동 프로파일:")
print(cluster_profile)


# ============================================
# 6) 그래프 자동 저장
# ============================================

# 1) 행동유형 분포 그래프
plt.figure()
df['behavior_type'].value_counts().sort_index().plot(kind='bar')
plt.title("Behavior Type Distribution")
plt.xlabel("Behavior Type")
plt.ylabel("Count")
plt.savefig("behavior_type_distribution.png", dpi=300)
plt.close()

# 2) 행동유형별 평균 피처 히트맵
plt.figure(figsize=(8,5))
sns.heatmap(cluster_profile, annot=True, cmap="Blues", fmt=".3f")
plt.title("Behavior Type Feature Profile")
plt.savefig("behavior_type_profile_heatmap.png", dpi=300)
plt.close()

print("\n 그래프 이미지 2개 저장 완료!")
print(" behavior_type_feature_profile.csv 저장 완료!")
