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
