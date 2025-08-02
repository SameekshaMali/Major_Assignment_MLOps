# src/utils.py

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error

def load_data():
    return fetch_california_housing(return_X_y=True)

def evaluate_model(model, X, y):
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    mse = mean_squared_error(y, preds)
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    return r2, mse

def save_params(obj, path):
    joblib.dump(obj, path)

def load_params(path):
    return joblib.load(path)
