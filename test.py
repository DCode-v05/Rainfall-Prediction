import os
import joblib  # For loading sklearn models
import pickle  # For loading CatBoost/XGBoost models
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Set paths
models_folder = "models/"  # Change to your models folder path
test_file = "train.csv"  # Change to your test file path

# Load test dataset
test_data = pd.read_csv(test_file)

# Assuming last column is the target variable
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Function to load models
def load_model(model_path):
    """Loads a model based on file extension."""
    if model_path.endswith(".pkl") or model_path.endswith(".joblib"):  
        return joblib.load(model_path)  # Sklearn models
    elif model_path.endswith(".cbm"):  
        from catboost import CatBoostClassifier  
        model = CatBoostClassifier()
        model.load_model(model_path)  # CatBoost models
        return model
    elif model_path.endswith(".json"):  
        import xgboost as xgb  
        return xgb.Booster(model_file=model_path)  # XGBoost models
    else:
        print(f"Skipping unknown model format: {model_path}")
        return None

# Loop through models in the folder
model_scores = {}
for model_file in os.listdir(models_folder):
    model_path = os.path.join(models_folder, model_file)
    
    model = load_model(model_path)
    if model is None:
        continue
    
    # Predict and evaluate
    if isinstance(model, xgb.Booster):  
        dtest = xgb.DMatrix(X_test)  # XGBoost format
        y_pred = np.round(model.predict(dtest))
    else:
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    model_scores[model_file] = accuracy

# Print results
print("\nModel Performance:")
for model_name, score in model_scores.items():
    print(f"{model_name}: {score:.4f}")

# Find the best model
best_model = max(model_scores, key=model_scores.get)
print(f"\nBest Model: {best_model} with accuracy {model_scores[best_model]:.4f}")


