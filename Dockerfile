# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

# Run training stage in build so model is baked in
ENV MODE=improved
ENV MODEL_VERSION=v0.2
RUN python train.py

# Final stage
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy app and baked model
COPY app ./app
COPY models ./models

EXPOSE 5000

ENV MODEL_PATH=models/model_v0.2.joblib
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app.api:app"]
