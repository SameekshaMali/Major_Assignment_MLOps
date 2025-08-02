# src/quantize.py

import numpy as np
import os
from utils import load_params, save_params, load_data
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def quantize_array(arr):
    min_val = arr.min()
    max_val = arr.max()
    scale = (max_val - min_val) / 255 if max_val != min_val else 1e-8
    q_arr = np.round((arr - min_val) / scale).astype(np.uint8)
    return q_arr, scale, min_val

def dequantize_array(q_arr, scale, min_val):
    return q_arr.astype(np.float32) * scale + min_val

def main():
    os.makedirs("models", exist_ok=True)

    model = load_params("models/model.joblib")
    scaler = load_params("models/scaler.joblib")
    weights = model.coef_
    bias = model.intercept_

    q_weights, scale, min_val = quantize_array(weights)
    save_params({
        "weights": q_weights,
        "scale": scale,
        "min_val": min_val,
        "bias": bias
    }, "models/quant_params.joblib")

    dq_weights = dequantize_array(q_weights, scale, min_val)

    X, y = load_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test) 

    preds = np.dot(X_test_scaled, dq_weights) + bias

    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    print("R2 after dequantization:", round(r2, 4))
    print("MSE after dequantization:", round(mse, 4))

if __name__ == "__main__":
    main()
