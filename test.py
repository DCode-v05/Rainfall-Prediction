import os
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

models_folder = "models/"
test_file = "train.csv"

test_data = pd.read_csv(test_file)

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

def load_model(model_path):
    """Loads a model based on file extension."""
    if model_path.endswith(".pkl") or model_path.endswith(".joblib"):  
        return joblib.load(model_path)
    elif model_path.endswith(".cbm"):  
        from catboost import CatBoostClassifier  
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model
    elif model_path.endswith(".json"):  
        import xgboost as xgb  
        return xgb.Booster(model_file=model_path)
    else:
        print(f"Skipping unknown model format: {model_path}")
        return None

model_scores = {}
for model_file in os.listdir(models_folder):
    model_path = os.path.join(models_folder, model_file)
    
    model = load_model(model_path)
    if model is None:
        continue
    
    if isinstance(model, xgb.Booster):  
        dtest = xgb.DMatrix(X_test)
        y_pred = np.round(model.predict(dtest))
    else:
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    model_scores[model_file] = accuracy

print("\nModel Performance:")
for model_name, score in model_scores.items():
    print(f"{model_name}: {score:.4f}")

best_model = max(model_scores, key=model_scores.get)
print(f"\nBest Model: {best_model} with accuracy {model_scores[best_model]:.4f}")

