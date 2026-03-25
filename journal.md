# Development Journal

## Week 1 — Jan 26 – Feb 1

Started the portfolio project. Decided on three topics: customer churn prediction, a computer vision/censorship task, and football match outcome prediction. Set up the Python environment using `uv`, added initial dependencies (pandas, scikit-learn, xgboost, matplotlib, seaborn), and created the project structure.

Began exploratory data analysis on the churn dataset (`customer_churn_train.csv`). Loaded the 594k-row dataset and checked for nulls, data types, and class balance. Found a 26% churn rate — moderately imbalanced. Plotted distributions of `tenure`, `MonthlyCharges`, and `TotalCharges`. Clear pattern: customers with low tenure and low total charges churn much more. Month-to-month contract customers also stand out — far higher churn rate than annual or two-year contracts.

---

## Week 2 — Feb 2 – Feb 8

Continued EDA on the churn notebook. Used mutual information scoring to rank all features against the `Churn` target. This helped narrow down from 20 columns to 10 meaningful features — 7 categorical and 3 numerical.

Set up a `ColumnTransformer` pipeline with `OneHotEncoder` for categoricals and `SimpleImputer` (median strategy) for numericals. Spent time getting the pipeline structure right so it could be reused cleanly for both baseline and clustering experiments later.

Trained the first baseline model — Random Forest with `class_weight='balanced'` to handle the imbalance. Got ROC-AUC of 0.887 on the validation set. Reasonable starting point.

---

## Week 3 — Feb 9 – Feb 15

Trained XGBoost as the second baseline model. Had to configure early stopping properly so it wouldn't overfit on the large dataset. Final ROC-AUC: 0.913 — noticeably better than Random Forest. XGBoost handles the mixed feature types and class imbalance better here.

Started thinking about the K-Means clustering experiment. The idea: cluster customers based on their numerical profile (tenure, monthly charges, total charges), then add the cluster label as an extra categorical feature and see if it helps the classifiers. Applied `StandardScaler` before clustering.

Ran the elbow curve for K=2–10. The inertia flattened around K=3. Calculated silhouette scores to confirm — K=3 gave the clearest separation (~0.4). Set K=3 and generated cluster assignments for the full training set.

---

## Week 4 — Feb 16 – Feb 22

Added the cluster column to the feature set and retrained both models. Results were anticlimactic: Random Forest dropped by 0.04% ROC-AUC, XGBoost gained 0.02%. Essentially neutral.

Made a PCA visualization to understand why — plotted the 3 clusters on the first two principal components and overlaid churn labels. The clusters loosely separated low-tenure, high-charge, and long-term customers, but the churn signal was spread across all three. Since the numerical features were already in the model, the cluster label wasn't adding new information.

Wrote up findings and moved on. Started the Premier League notebook. Downloaded 8 seasons of Premier League data from football-data.co.uk (2016/17 – 2023/24). Merged CSV files and did initial cleaning.

---

## Week 5 — Feb 23 – Mar 1

Focused entirely on feature engineering for the Premier League notebook. The core challenge: any features derived from the match itself (shots, corners, goals) can't be used to predict that same match — that's data leakage. Had to use rolling averages of past games as pre-match proxies.

Built a function to compute 5-game rolling averages per team per role (home/away separately), then applied `.shift(1)` so each row's features only include data from prior matches. Spent a while debugging this — the shift needs to happen before merging home and away stats, otherwise you get subtle leakage at the team boundary.

Checked for nulls introduced by the rolling window — 138 rows lost at the start of each team's season history. Acceptable.

---

## Week 6 — Mar 2 – Mar 8

Set up the temporal train/test split: 2016/17–2021/22 for training, 2022/23–2023/24 for testing. Important that this stays chronological — shuffling would let the model see future data during training, which would give falsely optimistic results and not reflect real betting conditions.

Noticed the test set has 58.8% Over 2.5 matches vs. 52.6% in training. That's a meaningful shift — likely reflects a genuine upward trend in Premier League scoring across those seasons.

Ran `GridSearchCV` for k-NN with 5-fold stratified cross-validation over 7 values of `n_neighbors`, 2 weight options, and 2 distance metrics. Best: n=5, distance weights, euclidean metric. CV ROC-AUC: 0.555.

---

## Week 7 — Mar 9 – Mar 15

Ran the SVM grid search — more computationally expensive due to the RBF kernel and gamma combinations. Best params: RBF kernel, C=1, gamma=0.01. CV ROC-AUC: 0.570. Slightly better than k-NN in cross-validation.

Evaluated both models on the test set. SVM clearly better: accuracy 0.556 vs 0.517, F1 0.686 vs 0.577, ROC-AUC 0.569 vs 0.510. Most striking difference is recall — SVM captures 82.5% of Over 2.5 matches vs. 55.9% for k-NN.

Built a betting simulation using €10 flat stakes. SVM returned +€424 (+5.7%). k-NN lost €127 (−1.7%). Calculated an "always Over" baseline for comparison — it returned +€880 (+11.8%) purely because the test set happened to have a high Over 2.5 rate. Sobering result. Wrote up reflections on why football prediction is fundamentally hard.

---

## Week 8 — Mar 16 – Mar 22

Started the censorship notebook. The goal: detect a middle finger in an image and draw a black rectangle over it. Decided on a three-stage pipeline: MediaPipe for hand landmark detection, feature extraction from the landmarks, and a classifier to identify the gesture.

Downloaded the MediaPipe `hand_landmarker.task` model. It predicts 21 3D keypoints per hand. Ran it on `hand_gesture.jpg` and plotted the landmarks to verify detection was working correctly.

Designed the feature representation: 5 finger extension ratios, computed as `(MCP_y - TIP_y) / hand_height`. Positive values mean the finger is extended, near-zero or negative mean it's curled. Scale-invariant since it's normalised by hand height. Implemented an `extract_features()` function.

Built the rule-based classifier first: middle ratio > 0.3 and all other fingers < 0.3. Tested on the image — detected correctly.

---

## Week 9 — Mar 23 – Mar 25

Trained a Keras MLP on top of the rule-based features. Generated 2,000 synthetic training samples — 1,000 positive (middle finger raised, other fingers curled) and 1,000 negative (random finger configurations excluding middle-up). Added Gaussian noise (σ=0.08) to simulate real variation.

Architecture: Input(5) → Dense(32, relu) → Dropout(0.2) → Dense(16, relu) → Dense(1, sigmoid). Trained for 40 epochs with Adam and binary cross-entropy. Validation accuracy converged to ~97%.

Ran the full pipeline on the test image: MediaPipe detects the hand, features are extracted, both the rule-based classifier and the Keras MLP flag it as a middle finger (MLP confidence 98.5%). Drew a black rectangle over landmarks 9–12 (middle finger MCP to TIP) with 3% padding.

Finished writing up all three notebooks and reviewed the portfolio as a whole.
