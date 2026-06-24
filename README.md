# Rainfall Prediction

**Binary "will it rain?" classification on daily weather data — two separate ML studies, seven models compared, with the trained classifiers saved to disk.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-006600?style=flat&logo=xgboost&logoColor=white) ![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat&logo=catboost&logoColor=black) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

## Overview

This repo holds two related machine-learning experiments, both answering the same kind of question: given a day's atmospheric readings, predict whether it rains (Yes/No). It's a tabular binary-classification project I built to practice the full supervised-learning loop — cleaning messy weather data, handling class imbalance, training and comparing several classifiers, and persisting the trained models.

There are two distinct notebooks, each working on a different dataset:

- A **Kaggle Playground-style competition** with a small, clean weather table (`train.csv` / `test.csv` / `sample_submission.csv`), where the goal is to produce a submission of predicted `rainfall` labels.
- A larger **"Rainfall in Australia" study** that predicts next-day rain from raw daily observations, comparing seven classifiers head to head.

The five `.pkl` files in `models/` are the trained classifiers from the Australia study. This is a learning project — the focus is the modelling workflow and the model comparison, not a deployed service.

## Key Features

- Two end-to-end classification pipelines on separate weather datasets, kept in their own notebooks.
- Missing-value handling with median imputation for numeric columns and mode imputation for categoricals.
- Categorical encoding (`get_dummies` / one-hot) for fields like `RainToday`, `RainTomorrow`, and wind-direction columns in the Australia study.
- Outlier capping by quantile clipping plus feature scaling (`StandardScaler`) in the competition pipeline.
- Class-imbalance correction with **SMOTE** oversampling (rain days are the minority class).
- Seven classifiers trained and compared: CatBoost, XGBoost, Random Forest, SVC, Logistic Regression, K-Nearest Neighbours, and Gaussian Naive Bayes.
- Hyperparameter tuning with `GridSearchCV` (5-fold) for Random Forest and XGBoost in the competition pipeline.
- Evaluation with accuracy, ROC-AUC, confusion matrices, and full classification reports, plus ROC-curve plots per model.
- Exploratory data analysis: target-distribution count plots, per-feature distributions, and correlation heatmaps.
- Five trained models serialized to `models/` (`cat.pkl`, `xgb.pkl`, `svc.pkl`, `logreg.pkl`, `gnb.pkl`) and a `test.py` helper that loads every saved model and reports accuracy.

## How It Works

The two notebooks share the same shape — clean the data, balance the classes, train several models, score them — but run on different data with slightly different steps.

### Pipeline A — Competition notebook (`Rainfall.ipynb`)

This works on the compact Kaggle-style table. `train.csv` has 2,190 rows and 12 numeric features (`day`, `pressure`, `maxtemp`, `temparature`, `mintemp`, `dewpoint`, `humidity`, `cloud`, `sunshine`, `winddirection`, `windspeed`, plus an `id`) and a binary `rainfall` target; `test.csv` has 730 unlabelled rows to score for `sample_submission.csv`.

1. **EDA** — target distribution count plot (the classes are imbalanced toward "rain"), per-feature distribution plots, and a correlation heatmap.
2. **Preprocessing** — quantile-based outlier capping on the numeric features, then `StandardScaler` so all features are on the same scale.
3. **Balancing** — `SMOTE` oversamples the minority class on the scaled features.
4. **Modelling** — an 80/20 stratified `train_test_split`, then Random Forest and XGBoost. Random Forest is tuned with `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf` (5-fold, scoring on accuracy); the best config came out as `max_depth=30, n_estimators=200, min_samples_split=2, min_samples_leaf=1`.
5. **Output** — predictions on `test.csv` formatted into the `id,rainfall` submission shape.

### Pipeline B — Australia study (`RainPrediction.ipynb`)

This is the classic "Rainfall in Australia" task: predict `RainTomorrow` from a day's raw observations. The dataset is much larger here — the held-out 20% test split alone is around 29,000 rows.

1. **Cleaning** — numeric gaps filled with the column median; categorical gaps (`WindGustDir`, `WindDir9am`, `WindDir3pm`, etc.) filled with the most frequent value. Binary text columns (`RainToday`, `RainTomorrow`) converted to 0/1 with `get_dummies(drop_first=True)`.
2. **Balancing** — `SMOTE` on the training set to even out rain vs. no-rain days.
3. **Modelling** — seven classifiers trained on the resampled data: **CatBoost** (`iterations=2000`, AUC eval metric), **XGBoost**, **Random Forest**, **SVC**, **Logistic Regression**, **KNN**, and **Gaussian Naive Bayes**.
4. **Evaluation** — each model scored with accuracy, a confusion matrix, a classification report, and a ROC-AUC value with its ROC curve plotted.
5. **Persistence** — the trained CatBoost, XGBoost, SVC, Logistic Regression, and Gaussian NB models saved as the `.pkl` files in `models/` (the `joblib.dump` calls are present in the notebook).

### Reloading saved models — `test.py`

`test.py` walks the `models/` folder, loads each serialized model (handling `.pkl`/`.joblib` via `joblib`, with branches for `.cbm` CatBoost and `.json` XGBoost formats), runs predictions, prints each model's accuracy, and reports the best one. Note it currently points its evaluation at `train.csv` (the 12-feature competition schema), which doesn't match the feature layout the saved Australia models were trained on — so it's a loading/scoring scaffold rather than a clean reproduction of the numbers below.

## Results / Highlights

From the executed `RainPrediction.ipynb` (Australia study, scored on the ~29k-row held-out test split). ROC-AUC is measured on the original imbalanced split, so it reads independently from accuracy:

| Model | Accuracy | ROC-AUC |
|---|---|---|
| CatBoost | 0.863 | 0.756 |
| XGBoost | 0.857 | 0.748 |
| Random Forest | 0.843 | 0.760 |
| SVC | 0.777 | 0.767 |
| Logistic Regression | 0.774 | 0.769 |
| KNN | 0.753 | 0.742 |
| Gaussian Naive Bayes | 0.749 | 0.746 |

CatBoost gives the highest raw accuracy (~86.3%); the linear/probabilistic models (Logistic Regression, SVC) edge ahead on ROC-AUC because they trade off some accuracy for better-ranked rain-day probabilities.

For the competition notebook (`Rainfall.ipynb`), the tuned Random Forest reached about **0.86** on its in-notebook validation split. That figure is on a SMOTE-balanced split, so treat it as a sanity-check number rather than a leaderboard-grade score.

## Tech Stack

- **Language:** Python 3
- **Notebook / analysis:** Jupyter Notebook
- **Data / ML:** pandas, NumPy, scikit-learn, XGBoost, CatBoost, imbalanced-learn (SMOTE)
- **Visualization:** Matplotlib, Seaborn
- **Model persistence:** joblib / pickle

## Getting Started

### Prerequisites
- Python 3.9+
- pip
- Jupyter (to open the notebooks)

### Installation
```bash
git clone https://github.com/DCode-v05/Rainfall-Prediction.git
cd Rainfall-Prediction
pip install pandas numpy scikit-learn xgboost catboost imbalanced-learn matplotlib seaborn jupyter joblib
```

### Running
```bash
# Open either notebook to step through the full pipeline
jupyter notebook Rainfall.ipynb        # competition pipeline (train/test/submission)
jupyter notebook RainPrediction.ipynb  # Australia study (7-model comparison)

# Load the saved models and print their accuracy
python test.py
```

## Usage

- To follow the modelling work, open the notebooks and run the cells top to bottom — they cover EDA, preprocessing, SMOTE balancing, training, and evaluation in order.
- To work with the pre-trained classifiers without retraining, point your code at the `.pkl` files in `models/` and load them with `joblib.load(...)`. Feed them feature rows in the same encoded layout the Australia notebook used.
- `test.py` is the quickest way to confirm all five saved models load and run; adjust its `test_file` path to a matching-schema dataset before reading its accuracy numbers.

## Project Structure

```
Rainfall-Prediction/
├── Rainfall.ipynb           # Competition pipeline: capping -> scaling -> SMOTE -> GridSearch RF/XGBoost
├── RainPrediction.ipynb     # Australia study: imputation -> SMOTE -> 7-model comparison + ROC curves
├── test.py                  # Loads every saved model from models/ and prints accuracy / best model
├── app.py                   # Stray pyautogui mouse snippet — not part of the ML pipeline
├── train.csv                # Competition training data (2,190 rows, 12 features + rainfall target)
├── test.csv                 # Competition test data (730 unlabelled rows)
├── sample_submission.csv    # Submission format: id,rainfall
├── models/                  # Serialized classifiers from the Australia study
│   ├── cat.pkl              # CatBoost
│   ├── xgb.pkl              # XGBoost
│   ├── svc.pkl              # Support Vector Classifier
│   ├── logreg.pkl           # Logistic Regression
│   └── gnb.pkl              # Gaussian Naive Bayes
└── README.md
```

---

## Contact

**Portfolio:** [Denistan](https://www.denistan.me)<br>
**LinkedIn:** [Denistan](https://www.linkedin.com/in/denistanb)<br>
**GitHub:** [DCode-v05](https://github.com/DCode-v05)<br>
**LeetCode:** [Denistan_B](https://leetcode.com/u/Denistan_B)<br>
**Email:** [denistanb05@gmail.com](mailto:denistanb05@gmail.com)

Made with ❤️ by **Denistan B**
