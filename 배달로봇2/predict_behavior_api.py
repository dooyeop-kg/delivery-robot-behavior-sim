from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# 모델 & 스케일러 불러오기
model = joblib.load("behavior_model.joblib")
scaler = joblib.load("scaler.joblib")

@app.get("/")
def home():
    return {"message": "Behavior Prediction API is running."}

@app.post("/predict")
def predict_behavior(data: dict):

    features = np.array([
        data["straightness"],
        data["stop_rate"],
        data["direction_change_rate"],
        data["traj_avg_speed"],
        data["path_length"],
        data["displacement"]
    ]).reshape(1, -1)

    # 스케일링
    scaled = scaler.transform(features)

    # 예측
    pred = model.predict(scaled)[0]
    probs = model.predict_proba(scaled)[0]

    return {
        "predicted_behavior": pred,
        "probabilities": probs.tolist()
    }
