import csv
import json
import subprocess
import requests
import time

def test_container_runs_and_predict(tmp_path):
    # This requires Docker to be installed in the test environment (CI).
    # Build the docker image
    tag = "test-model:latest"
    subprocess.run(["docker", "build", "-t", tag, "."], check=True)
    # Start container
    proc = subprocess.Popen(["docker", "run", "-p", "5001:5000", tag])
    time.sleep(5)  # wait for startup

    try:
        # Health endpoint
        r = requests.get("http://localhost:5001/health")
        assert r.status_code == 200
        j = r.json()
        assert j["status"] == "ok"

        # Predict endpoint
        payload = {k: 0.1 for k in ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]}
        r2 = requests.post("http://localhost:5001/predict", json=payload)
        assert r2.status_code == 200
        j2 = r2.json()
        assert "prediction" in j2
    finally:
        proc.terminate()
        proc.wait()
