# Data Science Portfolio 2

## 1. Premier League Over/Under 2.5 Goals

`premier_league.ipynb`

**Models:** SVM · K-Nearest Neighbours

8 seasons of Premier League data (3,040 matches). Rolling 5-game averages with lag-1 shifting are used as features to prevent data leakage, with a strict temporal train/test split to simulate real forecasting. Hyperparameters tuned via GridSearchCV. A betting simulation on the test set is included to evaluate practical value.

| Model | ROC-AUC | Betting P&L |
| ----- | ------- | ----------- |
| k-NN  | 0.510   | -€127       |
| SVM   | 0.569   | +€424       |

SVM's high recall (0.825) is the key driver of its profitability despite modest accuracy.

---

## 2. Customer Churn Prediction

`predict_customer_churn.ipynb`

**Models:** Random Forest · XGBoost

**K-Means clustering** (K=3) is applied to the numerical features to engineer a customer segment label, which is fed into two classifiers as an additional feature. Elbow curve and silhouette scoring were used to select K. The full preprocessing and modelling is wrapped in a single sklearn `Pipeline`.

| Model         | ROC-AUC |
| ------------- | ------- |
| Random Forest | 0.887   |
| XGBoost       | 0.914   |

Dataset: 594,194 telecom customers.

---

## 3. Middle Finger Censorship

`censorship.ipynb`

**Models:** MediaPipe BlazePalm CNN · Keras MLP

A two-stage neural network pipeline: MediaPipe detects 21 hand landmarks using a production-grade CNN (no training required), then a custom Keras MLP classifies the gesture from 5 engineered finger-extension ratio features. Detected gestures are censored in real time by drawing a black rectangle over the middle finger region using OpenCV. A rule-based classifier is also included as an interpretable baseline.
