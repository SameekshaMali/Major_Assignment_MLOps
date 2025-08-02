# src/train.py

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from utils import load_data, evaluate_model, save_params
from sklearn.preprocessing import StandardScaler
import os

def main():
    
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    os.makedirs("models", exist_ok=True)

    save_params(model, "models/model.joblib")
    save_params(scaler, "models/scaler.joblib")

    evaluate_model(model, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
