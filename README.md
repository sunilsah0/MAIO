# Virtual Diabetes Clinic Triage — MLOps Assignment

## Overview

This service predicts a short-term disease progression “risk” score for diabetes patients, to help triage nurse follow-ups.

It offers:

- `GET /health` → `{ "status": "ok", "model_version": "v0.1" }`  
- `POST /predict` → input JSON with features, returns `{ "prediction": <float> }`

We maintain two versions:

- **v0.1**: baseline (StandardScaler + LinearRegression)  
- **v0.2**: improved (Ridge / better preprocessing)

## Requirements & Setup

```bash
git clone <this repo>
cd repo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Run API server:

export MODEL_PATH=models/model_v0.2.joblib
python -m app.api


Or via Docker:

docker build -t mymodel:v0.2 .
docker run -p 5000:5000 mymodel:v0.2
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' \
     -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.
