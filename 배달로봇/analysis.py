import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np

# ============================
# 1) CSV 자동 불러오기
# ============================
# clean_A_10000_people.csv ~ clean_E_10000_people.csv 자동 탐지
file_pattern = "clean_*_10000_people.csv"
file_list = glob.glob(file_pattern)

print("불러올 파일 목록:", file_list)

# 파일 없으면 경고
if len(file_list) == 0:
    print("⚠ CSV 파일을 찾을 수 없습니다! 배달로봇 폴더에 clean_*.csv 파일을 넣어주세요.")
    exit()

# CSV 5개 읽기
all_data = []
for file in file_list:
    df_temp = pd.read_csv(file)
    all_data.append(df_temp)

# 하나의 df로 합치기
df = pd.concat(all_data, ignore_index=True)

print("데이터 불러오기 성공")
print(df.head())
import numpy as np

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
    if row['path_length'] == 0:
        return 0
    return row['displacement'] / row['path_length']

def compute_traj_avg_speed(row):
    return row['path_length'] / 9

def compute_stop_rate(row):
    return row['stop_count'] / 9

def compute_direction_change_rate(row):
    return row['direction_change'] / 9

df['path_length'] = df.apply(compute_path_length, axis=1)
df['displacement'] = df.apply(compute_displacement, axis=1)
df['straightness'] = df.apply(compute_straightness, axis=1)
df['traj_avg_speed'] = df.apply(compute_traj_avg_speed, axis=1)
df['stop_rate'] = df.apply(compute_stop_rate, axis=1)
df['direction_change_rate'] = df.apply(compute_direction_change_rate, axis=1)

print("파생 피처 생성 완료")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = ['path_length','displacement','straightness',
            'traj_avg_speed','stop_rate','direction_change_rate']

X = df[features]

# 스케일 조정
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# k=4로 클러스터링 (일단 추천)
kmeans = KMeans(n_clusters=4, random_state=42)
df['behavior_type'] = kmeans.fit_predict(X_scaled)

print(df['behavior_type'].value_counts())
loc_type_ratio = df.groupby(['location','behavior_type']).size().unstack(fill_value=0)
loc_type_ratio = loc_type_ratio.div(loc_type_ratio.sum(axis=1), axis=0)

print("\n장소별 행동 타입 비율:")
print(loc_type_ratio)

loc_type_ratio.to_csv("loc_behavior_ratio.csv")

# behavior_type 비율 시각화
df['behavior_type'].value_counts().sort_index().plot(kind='bar')
plt.title("Behavior Type Distribution")
plt.xlabel("Behavior Type")
plt.ylabel("Count")
plt.show()

import seaborn as sns

plt.figure(figsize=(8,5))
sns.heatmap(loc_type_ratio, annot=True, cmap="Blues", fmt=".3f")
plt.title("Location vs Behavior Type Ratio")
plt.show()

plt.scatter(df['straightness'], df['stop_rate'], alpha=0.3)
plt.title("Straightness vs Stop Rate")
plt.xlabel("Straightness")
plt.ylabel("Stop Rate")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# ────────────────────────────────
# 1. Behavior Type Distribution (막대그래프)
# ────────────────────────────────
plt.figure()
df['behavior_type'].value_counts().sort_index().plot(kind='bar')
plt.title("Behavior Type Distribution")
plt.xlabel("Behavior Type")
plt.ylabel("Count")
plt.savefig("behavior_type_distribution.png", dpi=300)   # PNG 저장
plt.close()

# ────────────────────────────────
# 2. 장소별 Behavior Type 비율 (Heatmap)
# ────────────────────────────────
plt.figure(figsize=(8,5))
sns.heatmap(loc_type_ratio, annot=True, cmap="Blues", fmt=".3f")
plt.title("Location vs Behavior Type Ratio")
plt.savefig("location_behavior_heatmap.png", dpi=300)
plt.close()

# ────────────────────────────────
# 3. Straightness vs Stop Rate (산점도)
# ────────────────────────────────
plt.figure()
plt.scatter(df['straightness'], df['stop_rate'], alpha=0.3)
plt.title("Straightness vs Stop Rate")
plt.xlabel("Straightness")
plt.ylabel("Stop Rate")
plt.savefig("straightness_vs_stoprate.png", dpi=300)
plt.close()

print(" 그래프 이미지 3개 저장 완료!")
