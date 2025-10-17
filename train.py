import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

RANDOM_SEED = 42

def load_data():
    Xy = load_diabetes(as_frame=True)
    df = Xy.frame
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def train_baseline(X_train, y_train):
    # simple pipeline
    pipeline = make_pipeline(StandardScaler(), LinearRegression())
    pipeline.fit(X_train, y_train)
    return pipeline

def train_improved(X_train, y_train):
    # for v0.2: try Ridge or RandomForest etc.
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_SEED))
    # Or: pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED))
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    return {"rmse": float(rmse)}

def main():
    mode = os.environ.get("MODE", "baseline")  # "baseline" or "improved"
    version = os.environ.get("MODEL_VERSION", "v0.1")

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    if mode == "improved":
        model = train_improved(X_train, y_train)
    else:
        model = train_baseline(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    print("Metrics:", metrics)

    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/model_{version}.joblib"
    joblib.dump(model, model_path)

    # Also write metrics file for CI to capture
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)

    # Optionally save the test set to later do integration tests
    X_test.to_csv("models/X_test.csv", index=False)
    y_test.to_csv("models/y_test.csv", index=False)

if __name__ == "__main__":
    main()
