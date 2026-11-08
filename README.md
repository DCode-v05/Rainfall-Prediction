# Rainfall Prediction

**Binary "will it rain?" classification on daily weather data: two separate ML studies on two datasets, with several models compared and some of the trained classifiers saved to disk.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-006600?style=flat&logo=xgboost&logoColor=white) ![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat&logo=catboost&logoColor=black) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

## Overview

This repo holds two related machine-learning experiments that ask the same kind of question: given a day's atmospheric readings, will it rain (Yes/No)? It's a tabular binary-classification project I built to practice the full supervised-learning loop: cleaning messy weather data, handling class imbalance, training and comparing several classifiers, and saving the trained models.

There are two notebooks, each on a different dataset:

- A **Kaggle Playground-style competition** with a small, clean weather table (`train.csv` / `test.csv` / `sample_submission.csv`), where the goal is a submission of predicted `rainfall` labels.
- A larger **"Rainfall in Australia" study** (`weatherAUS.csv`) that predicts next-day rain from raw daily observations and compares seven classifiers.

The five `.pkl` files in `models/` are trained classifiers from the Australia study. This is a learning project, so the focus is the modelling workflow and the model comparison, not a deployed service.

## Key Features

- Two classification pipelines on separate weather datasets, each in its own notebook.
- Missing-value handling for the Australia data: random-sample imputation for a few sparse columns, median for numeric columns and the most frequent value for categoricals.
- Categorical encoding: `get_dummies` for `RainToday` and `RainTomorrow`, and hand-written ordinal maps for the wind-direction and location columns.
- Outlier handling: IQR-based capping in the Australia study, and quantile clipping plus `StandardScaler` in the competition pipeline.
- Class-imbalance correction with **SMOTE** oversampling in both notebooks.
- Seven classifiers compared on the Australia data: CatBoost, XGBoost, Random Forest, SVC, Logistic Regression, K-Nearest Neighbours and Gaussian Naive Bayes.
- On the competition data: feature engineering and mutual-information feature selection, then Random Forest, XGBoost, a soft-voting mix of the two, a 1D CNN, an MLP and an ensemble of all four.
- `GridSearchCV` (5-fold) tuning for the Random Forest in the competition notebook.
- Evaluation with accuracy, ROC-AUC, confusion matrices and classification reports, plus ROC-curve plots per model in the Australia study.
- Exploratory data analysis: target-distribution count plots, per-feature distributions and correlation heatmaps.
- Five trained models in `models/` (`cat.pkl`, `xgb.pkl`, `svc.pkl`, `logreg.pkl`, `gnb.pkl`) and a `test.py` helper that loads each saved model and reports accuracy.

## How It Works

The two notebooks follow the same shape (clean the data, balance the classes, train several models, score them) but use different data and slightly different steps.

### Pipeline A — Competition notebook (`Rainfall.ipynb`)

This works on the compact Kaggle-style table. `train.csv` has 2,190 rows with an `id`, 11 numeric features (`day`, `pressure`, `maxtemp`, `temparature`, `mintemp`, `dewpoint`, `humidity`, `cloud`, `sunshine`, `winddirection`, `windspeed`) and a binary `rainfall` target. `test.csv` has 730 unlabelled rows to score for `sample_submission.csv`.

1. **EDA**: target distribution count plot (the classes are imbalanced toward "rain"), box plots, train-versus-test distributions and a correlation heatmap.
2. **Feature engineering and selection**: interaction and ratio features built from humidity, cloud and sunshine, then ranked with mutual information; ten features are kept.
3. **Preprocessing and balancing**: `StandardScaler`, `SMOTE` on the scaled features, then quantile clipping on a few columns.
4. **Modelling**: an 80/20 stratified split, then Random Forest, XGBoost, a soft-voting mix of both, a 1D CNN, an MLP and an ensemble of all four. Random Forest is tuned with `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_split` and `min_samples_leaf`; the best config was `max_depth=30, n_estimators=200, min_samples_split=2, min_samples_leaf=1`. An XGBoost grid search is set up in the notebook but no result is recorded.
5. **Output**: predictions on `test.csv` in the `id,rainfall` submission shape.

The notebook reads `/content/train.csv` and `/content/test.csv` (Colab paths), so change them to `train.csv` and `test.csv` to run it locally.

### Pipeline B — Australia study (`RainPrediction.ipynb`)

This is the classic "Rainfall in Australia" task: predict `RainTomorrow` from a day's raw observations. The held-out 20% test split alone is around 29,000 rows.

1. **Cleaning**: `Cloud9am`, `Cloud3pm`, `Evaporation` and `Sunshine` get random-sample imputation, remaining numeric gaps get the column median, and categorical gaps (`WindGustDir`, `WindDir9am`, `WindDir3pm`) get the most frequent value. `RainToday` and `RainTomorrow` become 0/1 with `get_dummies(drop_first=True)`, and the wind-direction and `Location` columns are mapped to integers with hand-written dictionaries. Outliers are capped with the IQR rule.
2. **Balancing**: `SMOTE` on the training set to even out rain and no-rain days.
3. **Modelling**: seven classifiers trained on the resampled data: **CatBoost** (`iterations=2000`, AUC eval metric), **XGBoost**, **Random Forest**, **SVC**, **Logistic Regression**, **KNN** (k=3) and **Gaussian Naive Bayes**.
4. **Evaluation**: each model scored with accuracy, a confusion matrix, a classification report, and a ROC-AUC value with its ROC curve plotted.
5. **Persistence**: the notebook's `joblib.dump` calls save `svc.pkl` and `xgb.pkl`. The `.pkl` files for CatBoost, Logistic Regression and Gaussian NB in `models/` come from the same study, but their dump calls are not in the notebook.

`weatherAUS.csv` is not in this repo, so this notebook needs the file from Kaggle.

### Reloading saved models — `test.py`

`test.py` walks the `models/` folder, loads each serialized model (`.pkl`/`.joblib` via `joblib`, with branches for `.cbm` CatBoost and `.json` XGBoost files), runs predictions, prints each model's accuracy and reports the best one. It has two problems. It evaluates on `train.csv`, which has the 12-column competition layout rather than the features the Australia models were trained on, so the scores it prints are not meaningful. It also uses `xgb` at module level without importing it (the import sits inside the `.json` branch), so it fails when it reaches an XGBoost model. Treat it as a loading scaffold that needs fixing, not a reproduction of the numbers below.

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

CatBoost has the highest accuracy (~86.3%). Logistic Regression and SVC come out ahead on ROC-AUC, since they trade some accuracy for better-ranked rain-day probabilities.

For the competition notebook (`Rainfall.ipynb`), validation accuracy from the notebook output is 0.853 for Random Forest, 0.855 for XGBoost and 0.861 for the Random Forest + XGBoost soft vote. These come from a SMOTE-balanced validation split, so treat them as sanity-check numbers rather than leaderboard scores.

## Tech Stack

- **Language:** Python 3
- **Notebook / analysis:** Jupyter Notebook
- **Data / ML:** pandas, NumPy, scikit-learn, XGBoost, CatBoost, imbalanced-learn (SMOTE), TensorFlow / Keras (CNN and MLP)
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
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn catboost xgboost tensorflow joblib jupyter
```

### Running
```bash
# Open either notebook to step through the full pipeline
jupyter notebook Rainfall.ipynb        # competition pipeline (train/test/submission)
jupyter notebook RainPrediction.ipynb  # Australia study (7-model comparison, needs weatherAUS.csv)

# Load the saved models and print their accuracy (see the caveats above first)
python test.py
```

## Usage

- To follow the modelling work, open the notebooks and run the cells top to bottom. They cover EDA, preprocessing, SMOTE balancing, training and evaluation in order. `Rainfall.ipynb` needs its Colab data paths changed, and `RainPrediction.ipynb` needs `weatherAUS.csv` downloaded from Kaggle.
- To use the pre-trained classifiers without retraining, load the `.pkl` files in `models/` with `joblib.load(...)` and feed them rows in the same encoded layout the Australia notebook used.
- `test.py` currently does not give meaningful numbers: point `test_file` at a dataset with the matching schema and fix the `xgb` import before relying on it.

## Project Structure

```
Rainfall-Prediction/
├── Rainfall.ipynb           # Competition pipeline: feature engineering -> SMOTE -> RF / XGBoost / CNN / MLP ensemble
├── RainPrediction.ipynb     # Australia study: imputation -> SMOTE -> 7-model comparison + ROC curves
├── test.py                  # Loads every saved model from models/ and prints accuracy / best model (needs fixes, see above)
├── train.csv                # Competition training data (2,190 rows, 12 columns incl. id + rainfall target)
├── test.csv                 # Competition test data (730 unlabelled rows)
├── sample_submission.csv    # Submission format: id,rainfall
├── models/                  # Serialized classifiers from the Australia study
│   ├── cat.pkl              # CatBoost
│   ├── xgb.pkl              # XGBoost
│   ├── svc.pkl              # Support Vector Classifier
│   ├── logreg.pkl           # Logistic Regression
│   └── gnb.pkl              # Gaussian Naive Bayes
├── .gitignore
└── README.md
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
