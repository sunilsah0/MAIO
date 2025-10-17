import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_v0.1.joblib")

class DiabetesModel:
    def __init__(self, model_path=None):
        path = model_path or MODEL_PATH
        self.pipeline: Pipeline = joblib.load(path)
        # You could capture a model_version property from filename or embed metadata.
        self.model_version = os.path.basename(path).replace("model_", "").replace(".joblib", "")

    def predict(self, features: dict) -> float:
        """
        features: mapping from feature names to numeric values.
        returns: float prediction (progression index)
        """
        # expected order: same as load_diabetes feature order
        feature_names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
        x = np.array([features[name] for name in feature_names]).reshape(1, -1)
        pred = self.pipeline.predict(x)[0]
        # Optionally clamp, calibrate, etc.
        return float(pred)
