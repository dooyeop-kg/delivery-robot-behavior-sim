import joblib
import pandas as pd

# ======================================
# 1) 스케일러 & 모델 불러오기
# ======================================
scaler = joblib.load("scaler.joblib")
model = joblib.load("behavior_model.joblib")

# 모델의 클래스(label) 순서 가져오기
class_labels = model.classes_


# ======================================
# 2) 단일 샘플 예측 함수
# ======================================
def predict_single(sample):

    feature_order = [
        "straightness",
        "stop_rate",
        "direction_change_rate",
        "traj_avg_speed",
        "path_length",
        "displacement"
    ]

    # pandas DataFrame으로 변환 → feature name 유지, 경고 제거
    df_sample = pd.DataFrame([sample], columns=feature_order)

    # 스케일 적용
    df_scaled = scaler.transform(df_sample)

    # 예측
    pred = model.predict(df_scaled)[0]

    # 예측 확률
    proba = model.predict_proba(df_scaled)[0]

    # 클래스별 확률 dict로 변환
    probability_dict = {label: round(prob, 4) for label, prob in zip(class_labels, proba)}

    return pred, probability_dict


# ======================================
# 3) 여러 샘플(batch) 예측 함수
# ======================================
def predict_batch(sample_list):

    feature_order = [
        "straightness",
        "stop_rate",
        "direction_change_rate",
        "traj_avg_speed",
        "path_length",
        "displacement"
    ]

    # DataFrame 생성
    df_samples = pd.DataFrame(sample_list, columns=feature_order)

    # 스케일
    df_scaled = scaler.transform(df_samples)

    # 예측
    preds = model.predict(df_scaled)

    # 확률
    probas = model.predict_proba(df_scaled)

    # 리스트 형태로 변환
    results = []
    for pred, proba in zip(preds, probas):
        probability_dict = {label: round(prob, 4) for label, prob in zip(class_labels, proba)}
        results.append({
            "predicted_behavior": pred,
            "probability": probability_dict
        })

    return results


# ======================================
# 4) 테스트 실행
# ======================================

# 단일 샘플 테스트
sample_input = {
    "straightness": 0.7,
    "stop_rate": 0.2,
    "direction_change_rate": 0.3,
    "traj_avg_speed": 1.10,
    "path_length": 8.5,
    "displacement": 3.1
}

pred, proba = predict_single(sample_input)

print("\n🔮 단일 샘플 예측 결과")
print("예측된 행동 타입:", pred)
print("각 행동 타입 확률:", proba)


# 여러 샘플 테스트
batch_samples = [
    {
        "straightness": 0.7,
        "stop_rate": 0.1,
        "direction_change_rate": 0.2,
        "traj_avg_speed": 1.3,
        "path_length": 10.5,
        "displacement": 4.2
    },
    {
        "straightness": 0.3,
        "stop_rate": 0.4,
        "direction_change_rate": 0.7,
        "traj_avg_speed": 0.8,
        "path_length": 6.2,
        "displacement": 2.1
    }
]

batch_results = predict_batch(batch_samples)

print("\n📦 배치 예측 결과")
for i, r in enumerate(batch_results):
    print(f"\n샘플 {i+1}:")
    print("예측된 행동:", r["predicted_behavior"])
    print("확률:", r["probability"])
