import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from main import app

client = TestClient(app)

@pytest.fixture
def sample_patient_data():
    return {
        "age": 65.0,
        "gender": 1.0,
        "bmi": 25.0,
        "disease_duration": 5.0,
        "updrs_score": 25.0,
        "jitter": 0.002,
        "shimmer": 0.015,
        "nhr": 0.02,
        "hnr": 20.0,
        "rpde": 0.5,
        "dfa": 0.7,
        "ppe": 0.4
    }

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Parkinson's Disease AI Clinical Decision Support System"}

def test_predict(sample_patient_data):
    response = client.post("/predict", json=sample_patient_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)
    assert 0 <= data["probability"] <= 1

def test_explain(sample_patient_data):
    response = client.post("/explain", json=sample_patient_data)
    assert response.status_code == 200
    data = response.json()
    assert "feature_importance" in data
    assert "shap_values" in data
    assert "lime_explanation" in data

def test_batch_predict():
    # Create a sample CSV file
    df = pd.DataFrame({
        "age": [65.0, 70.0],
        "gender": [1.0, 0.0],
        "bmi": [25.0, 28.0],
        "disease_duration": [5.0, 8.0],
        "updrs_score": [25.0, 30.0],
        "jitter": [0.002, 0.003],
        "shimmer": [0.015, 0.018],
        "nhr": [0.02, 0.025],
        "hnr": [20.0, 18.0],
        "rpde": [0.5, 0.6],
        "dfa": [0.7, 0.8],
        "ppe": [0.4, 0.5]
    })
    
    # Save to temporary file
    csv_file = "test_data.csv"
    df.to_csv(csv_file, index=False)
    
    # Test batch prediction
    with open(csv_file, "rb") as f:
        response = client.post("/batch-predict", files={"file": ("test_data.csv", f, "text/csv")})
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    
    # Clean up
    import os
    os.remove(csv_file)

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "feature_importance" in data
    assert "model_metrics" in data 