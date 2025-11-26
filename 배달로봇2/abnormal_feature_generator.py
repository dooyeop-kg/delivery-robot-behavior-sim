import numpy as np
import pandas as pd

# =========================================================
#  이상행동 7가지 정의 (A안)
# =========================================================

def gen_zigzag(n=500):
    return pd.DataFrame({
        "behavior": ["zigzag"] * n,
        "straightness": np.random.uniform(0.1, 0.4, n),
        "stop_rate": np.random.uniform(0.05, 0.15, n),
        "direction_change_rate": np.random.uniform(0.4, 0.8, n),
        "traj_avg_speed": np.random.uniform(0.8, 1.4, n),
        "path_length": np.random.uniform(8, 12, n),
        "displacement": np.random.uniform(2, 5, n)
    })

def gen_frequent_stop(n=500):
    return pd.DataFrame({
        "behavior": ["frequent_stop"] * n,
        "straightness": np.random.uniform(0.3, 0.7, n),
        "stop_rate": np.random.uniform(0.4, 0.9, n),
        "direction_change_rate": np.random.uniform(0.1, 0.3, n),
        "traj_avg_speed": np.random.uniform(0.3, 0.8, n),
        "path_length": np.random.uniform(5, 9, n),
        "displacement": np.random.uniform(1, 4, n)
    })

def gen_running(n=500):
    return pd.DataFrame({
        "behavior": ["running"] * n,
        "straightness": np.random.uniform(0.8, 1.0, n),
        "stop_rate": np.random.uniform(0.0, 0.05, n),
        "direction_change_rate": np.random.uniform(0.0, 0.1, n),
        "traj_avg_speed": np.random.uniform(1.8, 3.0, n),
        "path_length": np.random.uniform(12, 18, n),
        "displacement": np.random.uniform(10, 16, n)
    })

def gen_slow_wandering(n=500):
    return pd.DataFrame({
        "behavior": ["slow_wandering"] * n,
        "straightness": np.random.uniform(0.2, 0.6, n),
        "stop_rate": np.random.uniform(0.1, 0.3, n),
        "direction_change_rate": np.random.uniform(0.15, 0.4, n),
        "traj_avg_speed": np.random.uniform(0.2, 0.6, n),
        "path_length": np.random.uniform(4, 7, n),
        "displacement": np.random.uniform(1, 3, n)
    })

def gen_backward(n=500):
    return pd.DataFrame({
        "behavior": ["backward"] * n,
        "straightness": np.random.uniform(0.7, 1.0, n),
        "stop_rate": np.random.uniform(0.05, 0.2, n),
        "direction_change_rate": np.random.uniform(0.05, 0.2, n),
        "traj_avg_speed": np.random.uniform(0.6, 1.2, n),
        "path_length": np.random.uniform(6, 10, n),
        "displacement": np.random.uniform(-5, -1, n)  # 음수: 뒤로 이동
    })

def gen_circling(n=500):
    return pd.DataFrame({
        "behavior": ["circling"] * n,
        "straightness": np.random.uniform(0.0, 0.2, n),
        "stop_rate": np.random.uniform(0.05, 0.15, n),
        "direction_change_rate": np.random.uniform(0.7, 1.0, n),
        "traj_avg_speed": np.random.uniform(0.5, 1.0, n),
        "path_length": np.random.uniform(6, 10, n),
        "displacement": np.random.uniform(0.5, 2, n)  # 제자리 근처 이동
    })

def gen_random(n=500):
    return pd.DataFrame({
        "behavior": ["random"] * n,
        "straightness": np.random.uniform(0.0, 0.3, n),
        "stop_rate": np.random.uniform(0.0, 0.2, n),
        "direction_change_rate": np.random.uniform(0.3, 1.0, n),
        "traj_avg_speed": np.random.uniform(0.4, 1.4, n),
        "path_length": np.random.uniform(5, 11, n),
        "displacement": np.random.uniform(1, 7, n)
    })


# =========================================================
#   7종 데이터 합치기
# =========================================================
df_final = pd.concat([
    gen_zigzag(),
    gen_frequent_stop(),
    gen_running(),
    gen_slow_wandering(),
    gen_backward(),
    gen_circling(),
    gen_random()
], ignore_index=True)

# 저장
df_final.to_csv("abnormal_features.csv", index=False)

print("\n 이상행동 7종 데이터 생성 완료!")
print("파일 생성: abnormal_features.csv")
print(df_final.head())
