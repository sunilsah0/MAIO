from flask import Flask, request, jsonify
from .model import DiabetesModel

app = Flask(__name__)
model = DiabetesModel()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_version": model.model_version})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        # Validate keys
        required = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
        if payload is None:
            raise ValueError("Empty JSON payload")
        for key in required:
            if key not in payload:
                raise ValueError(f"Missing feature: {key}")
        pred = model.predict(payload)
        return jsonify({"prediction": pred})
    except Exception as e:
        response = {"error": str(e)}
        return jsonify(response), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
