# Teen Mental Health Risk Prediction using SVM

## Project Overview

This project predicts adolescent depression risk using the **Teen Social Media Usage & Mental Health** dataset from Kaggle.

A Support Vector Machine (SVM) with an RBF kernel was trained to classify depression risk into three categories:

- Low
- Medium
- High

---

## Dataset

**Dataset Name**

Teen Social Media Usage & Mental Health

**Source**

https://www.kaggle.com/datasets/sureshbeekhani/teen-social-media-usage-and-mental-health

> **Note**
>
> Download `Teen_Mental_Health_Dataset.csv` from the Kaggle dataset above and place it in the same directory as `healthmental.m` before running the code.

---

## Features

The following features were used for training.

- Daily Social Media Usage Time
- Stress Level
- Anxiety Level
- Sleep Hours
- Academic Performance (GPA)

**Target Variable**

- Depression Risk
  - Low
  - Medium
  - High

---

## Machine Learning Model

- Support Vector Machine (SVM)
- RBF Kernel
- One-vs-One Multi-class Classification

---

## Results

- Correlation Heatmap
- Scatter Plot
- Confusion Matrix
- ROC Curve
- Accuracy
- Macro AUC

---

## Project Structure

```
SVM-Classification/
│
├── README.md
├── svm_classification.m
```

---

## How to Run

1. Download the dataset from Kaggle.
2. Extract `Teen_Mental_Health_Dataset.csv`.
3. Place the CSV file in the same folder as `svm_classification.m`.
4. Open MATLAB.
5. Run `svm_classification.m`.

---

## Programming Language

- MATLAB

---

## Author

Gyeong Ho Min
