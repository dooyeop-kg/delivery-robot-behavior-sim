import numpy as np

# 1) path_length (총 이동 거리)
def compute_path_length(row):
    dist = 0
    for i in range(9):
        x1, y1 = row[f'x_{i}'], row[f'y_{i}']
        x2, y2 = row[f'x_{i+1}'], row[f'y_{i+1}']
        dist += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

# 2) displacement (시작~끝 직선 거리)
def compute_displacement(row):
    return np.sqrt((row['x_9'] - row['x_0'])**2 + (row['y_9'] - row['y_0'])**2)

# 3) straightness = displacement / path_length
def compute_straightness(row):
    if row['path_length'] == 0:
        return 0
    return row['displacement'] / row['path_length']

# 4) traj_avg_speed = path_length / 총 프레임 시간
# 프레임이 10개 → 9개 구간 → 9초로 가정
def compute_traj_avg_speed(row):
    return row['path_length'] / 9

# 5) stop_rate = stop_count / 9
def compute_stop_rate(row):
    return row['stop_count'] / 9

# 6) direction_change_rate = direction_change / 9
def compute_direction_change_rate(row):
    return row['direction_change'] / 9


# =====================
#  전체 피처 생성
# =====================
df['path_length'] = df.apply(compute_path_length, axis=1)
df['displacement'] = df.apply(compute_displacement, axis=1)
df['straightness'] = df.apply(compute_straightness, axis=1)
df['traj_avg_speed'] = df.apply(compute_traj_avg_speed, axis=1)
df['stop_rate'] = df.apply(compute_stop_rate, axis=1)
df['direction_change_rate'] = df.apply(compute_direction_change_rate, axis=1)

df[['path_length','displacement','straightness','traj_avg_speed','stop_rate','direction_change_rate']].head()
