# Rainfall Prediction

Two notebooks that predict whether it will rain from daily weather readings, on two different datasets, plus a script that loads the saved models.

## Notebooks

**`RainPrediction.ipynb`** works on the Australian weather data (`weatherAUS.csv`, target `RainTomorrow`). It cleans and encodes the data, handles outliers, balances the training set with SMOTE, and trains seven models on an 80/20 stratified split. Accuracy on the test split, from the notebook output:

| Model | Accuracy |
|---|---|
| CatBoost | 0.863 |
| XGBoost | 0.857 |
| Random Forest | 0.843 |
| SVC | 0.777 |
| Logistic Regression | 0.774 |
| KNN (k=3) | 0.753 |
| Gaussian Naive Bayes | 0.749 |

`weatherAUS.csv` is not in this repo, so this notebook needs the file from Kaggle. The models in `models/` (`cat.pkl`, `xgb.pkl`, `svc.pkl`, `logreg.pkl`, `gnb.pkl`) come from this notebook.

**`Rainfall.ipynb`** is a binary-classification task on `train.csv` (2,190 days with pressure, temperatures, dew point, humidity, cloud, sunshine, wind and the `rainfall` target) and `test.csv`. `sample_submission.csv` gives the submission format. It checks class imbalance and outliers, does some feature engineering and selection, then tries Random Forest, XGBoost, a soft-voting mix of the two, a 1D CNN and an MLP, and finally an ensemble of all four. Validation accuracy from the output: Random Forest 0.853, XGBoost 0.855, Random Forest plus XGBoost 0.861.

## test.py

Loads every file in `models/` and prints its accuracy. It points at `train.csv`, which has a different schema from the weatherAUS data the models were trained on, so the scores it prints are not meaningful. It also uses `xgb` without importing it at module level, so it fails when it reaches the XGBoost branch. Both need fixing before it is useful.

## Running it

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn catboost xgboost tensorflow joblib jupyter
jupyter notebook
```

`Rainfall.ipynb` reads `/content/train.csv` and `/content/test.csv` (Colab paths); change them to `train.csv` and `test.csv`.
