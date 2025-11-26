import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib

print("📂 final_behavior_dataset.csv 불러오는 중...")

df = pd.read_csv("final_behavior_dataset.csv", low_memory=False)

print(df.head())
print("\n📌 데이터 컬럼 목록:")
print(list(df.columns))

# ================================
# 1) 컬럼 매핑 (데이터셋 → 모델용 표준 컬럼명)
# ================================
column_map = {
    "straightness": "straightness",
    "stop_rate_per_s": "stop_rate",
    "direction_change_rate": "direction_change_rate",
    "traj_avg_speed_ms": "traj_avg_speed",
    "path_length_m": "path_length",
    "displacement_m": "displacement"
}

df = df.rename(columns=column_map)

required = [
    "straightness", 
    "stop_rate",
    "direction_change_rate",
    "traj_avg_speed",
    "path_length",
    "displacement"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"❌ 매핑 후에도 존재하지 않는 컬럼이 있습니다: {missing}")

# ================================
# 2) behavior_type 문자열로 통일
# ================================
df["behavior_type"] = df["behavior_type"].astype(str)

print("\n✔ 통합된 behavior_type 값:")
print(df["behavior_type"].value_counts())

# ================================
# 3) 특성과 라벨 분리
# ================================
X = df[required]
y = df["behavior_type"]

# ================================
# 4) 데이터 분할
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ================================
# 5) 스케일링
# ================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.joblib")
print("\n💾 scaler.joblib 저장 완료!")

# ================================
# 6) 모델 학습
# ================================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# ================================
# 7) 평가
# ================================
pred = model.predict(X_test_scaled)

print("\n=== 혼동 행렬 ===")
print(confusion_matrix(y_test, pred))

print("\n=== 분류 리포트 ===")
print(classification_report(y_test, pred))

# ================================
# 8) 모델 저장
# ================================
joblib.dump(model, "behavior_model.joblib")
print("\n🔥 모델 학습 및 저장 완료: behavior_model.joblib")
