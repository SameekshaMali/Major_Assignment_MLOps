# tests/test_train.py

import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from src.utils import load_data, evaluate_model

def test_model_training():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    r2, mse = evaluate_model(model, X_test, y_test)

    assert r2 > 0.4, f"R2 score too low: {r2}"
    assert mse < 2, f"MSE too high: {mse}"
