# Rainfall Prediction

## Project Description
This project focuses on predicting rainfall using machine learning techniques. Leveraging the "Rainfall in Australia" dataset (or similar meteorological data), it builds binary classification models to predict whether it will rain tomorrow based on various atmospheric features like temperature, humidity, pressure, and wind speed.

---

## Project Details

### Problem Statement
Accurate rainfall prediction is vital for agriculture, water resource management, and daily planning. This project aims to develop robust classification models that can analyze daily weather observations and accurately forecast rain (Yes/No).

### Data Preprocessing
- **Data Source:** The project uses `train.csv` and `test.csv` containing meteorological data.
- **Features:** Columns include MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed, Humidity, Pressure, Cloud, etc.
- **Handling Missing Values:** Imputation strategies (e.g., mean/mode or random sample imputation) are used to handle missing entries in columns like Cloud, Evaporation, and Sunshine.
- **Encoding:** Categorical variables (e.g., Location, WindDir) are encoded for model training.

### Model Training & Evaluation
Several popular machine learning algorithms were trained and evaluated:
- **CatBoost Classifier** (`cat.pkl`)
- **XGBoost Classifier** (`xgb.pkl`)
- **Support Vector Classifier (SVC)** (`svc.pkl`)
- **Logistic Regression** (`logreg.pkl`)
- **Gaussian Naive Bayes** (`gnb.pkl`)

The models were evaluated based on accuracy and other metrics to select the best performer for this classification task.

### Visualizations
The analysis notebooks include:
- Distribution plots for numerical features.
- Correlation heatmaps to understand feature relationships.
- Analysis of feature importance.

---

## Tech Stack
- **Python 3.x**
- **Pandas, NumPy** (Data Manipulation)
- **Scikit-learn** (Modeling & Evaluation)
- **XGBoost, CatBoost** (Gradient Boosting Models)
- **Matplotlib, Seaborn** (Visualization)
- **Jupyter Notebook** (Analysis)

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/DCode-v05/Rainfall-Prediction.git
cd "Rainfall Prediction"
```

### 2. Install dependencies
Ensure you have the required libraries installed:
```bash
pip install pandas numpy scikit-learn xgboost catboost matplotlib seaborn jupyter
```

### 3. Run the Analysis
To view the data analysis and training process, run the main notebook:
```bash
jupyter notebook Rainfall.ipynb
```
*Note: `RainPrediction2.ipynb` contains additional/alternative analysis.*

### 4. Evaluate Saved Models
A script is provided to load trained models from the `models/` directory and evaluate them against the training set:
```bash
python temp.py
```

---

## Project Structure
```
Rainfall Prediction/
│
├── Rainfall.ipynb          # Main analysis and training notebook
├── RainPrediction.ipynb    # Alternative/Draft analysis notebook
├── test.py                 # Script to evaluate saved models
├── app.py                  # Utility script (mouse automation)
├── models/                 # Directory containing seralized models
│   ├── cat.pkl             # CatBoost model
│   ├── gnb.pkl             # Gaussian Naive Bayes model
│   ├── logreg.pkl          # Logistic Regression model
│   ├── svc.pkl             # Support Vector Classifier model
│   └── xgb.pkl             # XGBoost model
├── train.csv               # Training dataset
├── test.csv                # Test dataset
├── sample_submission.csv   # Sample submission file
└── README.md               # Project documentation
```

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

---

## Contact
- **GitHub:** [DCode-v05](https://github.com/DCode-v05)
- **Email:** denitanb05@gmail.com
