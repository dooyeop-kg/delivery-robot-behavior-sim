import pandas as pd

# ============================
# 1. 정상 데이터 불러오기
# ============================
normal_df = pd.read_csv("combined_features_with_behavior.csv")
print("정상 데이터 컬럼:", list(normal_df.columns))


# ============================
# 2. 이상 데이터 불러오기
# ============================
abnormal_df = pd.read_csv("abnormal_features.csv")
print("\n이상 데이터 컬럼:", list(abnormal_df.columns))


# ============================
# 3. 컬럼 이름을 정상 데이터에 맞게 수정
# ============================

rename_map = {
    "path_length": "path_length_m",
    "displacement": "displacement_m",
    "traj_avg_speed": "traj_avg_speed_ms",
    "stop_rate": "stop_rate_per_s",
    "direction_change_rate": "direction_change_rate",
    "straightness": "straightness",
}

abnormal_df = abnormal_df.rename(columns=rename_map)

# behavior → behavior_type 으로 변경 (정상 데이터와 동일하게)
abnormal_df["behavior_type"] = abnormal_df["behavior"].astype(str)
abnormal_df = abnormal_df.drop(columns=["behavior"])


# ============================
# 4. 정상 DF에는 있는데, 이상 DF에 없는 컬럼 추가
# ============================
normal_cols = list(normal_df.columns)

for col in normal_cols:
    if col not in abnormal_df.columns:
        abnormal_df[col] = None   # 결측값으로 채움 → 시뮬레이터에서 의미 있는 처리 가능


# ============================
# 5. 컬럼 순서 동일하게 맞추기
# ============================
abnormal_df = abnormal_df[normal_cols]


# ============================
# 6. 정상 + 이상 병합
# ============================
final_df = pd.concat([normal_df, abnormal_df], ignore_index=True)

print("\n병합 완료! 전체 행 개수:", len(final_df))

final_df.to_csv("final_behavior_dataset.csv", index=False)
print("✔ 최종 파일 저장 완료: final_behavior_dataset.csv")
