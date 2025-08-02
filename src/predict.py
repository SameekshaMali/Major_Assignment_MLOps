# src/predict.py

from utils import load_params, load_data
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    model = load_params("models/model.joblib")
    scaler = load_params("models/scaler.joblib")

    X, y = load_data()
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)

    preds = model.predict(X_test_scaled)

    print("Sample Predictions (first 5):")
    for i, val in enumerate(preds[:5]):
        print(f"{i+1}. {val:.4f}")

if __name__ == "__main__":
    main()
