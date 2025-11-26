# 🚚 배달로봇 군중 행동 패턴 분석 & 시뮬레이션용 행동 데이터 생성 프로젝트  

본 프로젝트는 배달로봇의 **안전한 이동 경로 계획과 군중 회피 동작 구현**을 위해  
실제 보행자 센서 기반 이동 데이터를 분석하고 행동 타입(0~7)을 분류하여  
시뮬레이터에서 **바로 활용 가능한 행동 데이터(JSON/CSV, 예측 API)** 를 생성한  
**End-to-End AI 프로젝트**입니다.

---

## 📌 프로젝트 목표

- 보행자 행동 타입(type_0~7) 정의 및 분류 모델 생성  
- 실제 데이터 기반 Feature Engineering 및 품질 검증  
- 정상/이상 행동 합성 데이터 생성  
- 배달로봇 시뮬레이터에 바로 투입 가능한 행동 데이터셋 구축  
- FastAPI 기반 실시간 예측 기능 제공  

---

## 🛠 기술 스택

### 언어 & 라이브러리
- Python  
- Pandas, Numpy  
- Scikit-learn (RandomForest)  

### 시스템 / 서비스
- FastAPI + Uvicorn  
- Git / GitHub (형상 관리)  

---

## 📁 프로젝트 구조
```
delivery-robot-behavior-sim/
│
├─ abnormal_features.csv # 이상 행동 Feature 데이터
├─ combined_features_with_behavior.csv # 정상+이상 병합 데이터
├─ final_behavior_dataset.csv # 최종 학습용 데이터셋
│
├─ behavior_model_training.py # ML 모델 학습 코드
├─ behavior_model.joblib # 학습된 RandomForest 모델
│
├─ predict_behavior.py # 단일 보행자 예측 코드
├─ predict_behavior_api.py # FastAPI 기반 예측 API
├─ scaler.joblib # StandardScaler 저장본
│
├─ merge_normal_abnormal.py # 정상+이상 데이터 병합
├─ feature_analysis.py # EDA / Feature Engineering 분석
│
└─ README.md # 프로젝트 설명 문서
```

---

## 📊 데이터 설명

### 원본 데이터 특징

- 약 **50,000건 이상**의 보행자 이동 데이터  
- 단독 또는 2인 보행자 비중 높음 (약 83%)  
- 센서 기반 X,Y 좌표 시퀀스 데이터  

### 주요 변수

- speed  
- straightness  
- stop_rate  
- direction_change_rate  
- path_length  
- displacement  
- traj_avg_speed  

---

## 🧬 Feature Engineering

### Feature 설명

| Feature               | 설명                             |
|-----------------------|----------------------------------|
| path_length           | X,Y 좌표 기반 이동 거리          |
| displacement          | 시작점 ↔ 끝점 직선거리           |
| straightness          | displacement / path_length       |
| stop_rate             | 정지 구간 비율                   |
| direction_change_rate | 방향 변화량 기반 회전성          |
| traj_avg_speed        | 평균 속도                         |

✔ **DE 팀 제공 공식 Feature + 직접 생성 Feature가 결합된 확장형 Feature Set 사용**

---

## 🎯 행동 타입 정의 (0~7)

| 타입 | 의미            |
|------|-----------------|
| 0    | Running         |
| 1    | Walking         |
| 2    | Circling        |
| 3    | Backward        |
| 4    | Frequent Stop   |
| 5    | Random          |
| 6    | Zigzag          |
| 7    | Slow Wandering  |

---

## 🤖 모델 학습 (RandomForest Classifier)

### 사용 모델

- RandomForestClassifier  
- StandardScaler 적용  
- GridSearch 기반 최적 파라미터 검색 가능 (추후 확장)  

### 학습 코드 파일

behavior_model_training.py



### 주요 성능 지표

- Confusion Matrix 제공  
- Class별 예측 확률 출력  
- (확장 예정) Macro F1 Score, Recall  

---

## 🎯 예측 코드 사용법

### 1) 단일 보행자 예측 실행

python predict_behavior.py


#### 입력 예시

{
"straightness": 0.7,
"stop_rate": 0.2,
"direction_change_rate": 0.3,
"traj_avg_speed": 1.1,
"path_length": 8.5,
"displacement": 3.1
}


#### 출력 예시

{
"predicted_behavior": "zigzag",
"probabilities": {
"0": 0.01,
"1": 0.13,
"2": 0.04,
"3": 0.06,
"4": 0.20,
"5": 0.07,
"6": 0.37,
"7": 0.11
}
}



---

## 🌐 FastAPI 실시간 예측 서버

### 서버 실행

uvicorn predict_behavior_api:app --reload


### Swagger UI

- http://127.0.0.1:8000/docs  

### 예측 엔드포인트

- `POST /predict`

---

## 🔄 1차 vs 2차 작업 차이

### 🚀 1차 버전

- 직접 Feature Engineering 수행  
- **정상 행동 중심** 데이터  
- 단일 행동 타입 분류 모델 구축  
- 시뮬레이터에 투입 가능한 **베이스라인 모델** 완성  

### 🚀 2차 버전 (현재 최종 버전)

- **DE 팀 제공 공식 Feature** 적용  
- 데이터 품질 검사 및 정상 범위 검증 완료  
- **정상 + 이상 행동 데이터 병합**  
- 모델 정확도·일관성·확장성 강화  
- FastAPI 기반 **실사용 예측 서비스** 구현  

➡ 1차 대비 **품질·현실성·신뢰도**가 크게 향상된 버전입니다.

---

## 📈 향후 확장 방향

- 시간대별 / 연령별 / 장소별 군중 행동 모델링  
- 강화학습(RL) 기반 로봇 회피 동작 연동  
- 보행자 간 상호작용 Feature 추가 (근접도, 상대 속도 등)  
- 시뮬레이터와 End-to-End 자동 파이프라인 구축  
- 대규모 데이터 기반 Transformer 계열 모델로 확장  

---

## ✔ 프로젝트 한줄 정리

**데이터 분석 → Feature Engineering → 모델 학습 → 예측 API 배포**  
까지 모두 구현한, 배달로봇·군중 시뮬레이션 도메인에 특화된 **End-to-End 행동 예측 프로젝트**입니다.

---

## 🔗 개발자 정보

- GitHub: https://github.com/dooyeop-kg  
- Email: rlaenduqv@naver.com
