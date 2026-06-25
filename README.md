# 청소년 소셜미디어 사용과 정신건강 분석

## 프로젝트 소개

본 프로젝트는 Kaggle의 Teen Social Media Usage & Mental Health 데이터를 이용하여
청소년의 우울 위험도를 예측하는 머신러닝 프로젝트입니다.

Support Vector Machine(SVM, RBF Kernel)을 이용하여
Low / Medium / High 위험군을 분류하였습니다.

---

## 사용 데이터

- Dataset : Teen Social Media Usage & Mental Health (Kaggle)
- 약 2,500명의 청소년 데이터

### 입력 변수

- SNS Usage Time
- Sleep Time
- Stress Level
- Anxiety Level
- GPA

### 목표 변수

- Depression Risk
  - Low
  - Medium
  - High

---

## 사용 기술

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 진행 과정

1. 데이터 전처리
2. 상관관계 분석
3. Train / Test Split
4. SVM(RBF) 모델 학습
5. Confusion Matrix 분석
6. ROC Curve 평가

---

## 결과

- Macro AUC : 0.905
- High Risk Recall : 95%
- Low Risk Recall : 82.5%
- Medium Risk Recall : 53.2%

---

## 프로젝트 자료

발표 PPT 포함

---

## 작성자

Gyeong Ho Min
