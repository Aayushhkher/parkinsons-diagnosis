from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import io

from ml.features.engineer import FeatureEngineer
from ml.models.train import ModelTrainer
from ml.explainability.explainer import ModelExplainer

app = FastAPI(title="Parkinson's Disease AI Clinical Decision Support System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
feature_engineer = FeatureEngineer(n_components=0.95)
model_trainer = ModelTrainer(model_type='ensemble')

# Train the model with some sample data
sample_data = pd.DataFrame({
    'age': [65.0, 70.0, 75.0],
    'gender': [1.0, 0.0, 1.0],
    'bmi': [25.0, 28.0, 26.0],
    'disease_duration': [5.0, 8.0, 10.0],
    'updrs_score': [25.0, 30.0, 35.0],
    'jitter': [0.002, 0.003, 0.004],
    'shimmer': [0.015, 0.018, 0.02],
    'nhr': [0.02, 0.025, 0.03],
    'hnr': [20.0, 18.0, 16.0],
    'rpde': [0.5, 0.6, 0.7],
    'dfa': [0.7, 0.8, 0.9],
    'ppe': [0.4, 0.5, 0.6]
})
sample_labels = np.array([0, 1, 1])

# Preprocess and train
X_processed = feature_engineer.engineer_features(sample_data)
model_trainer.train(X_processed, sample_labels)

# Initialize model explainer
model_explainer = ModelExplainer(
    model=model_trainer.rf_model,
    feature_names=X_processed.columns.tolist()
)

class PatientData(BaseModel):
    age: float
    gender: float
    bmi: float
    disease_duration: float
    updrs_score: float
    jitter: float
    shimmer: float
    nhr: float
    hnr: float
    rpde: float
    dfa: float
    ppe: float

@app.get("/")
async def root():
    return {"message": "Welcome to Parkinson's Disease AI Clinical Decision Support System"}

@app.post("/predict")
async def predict(data: PatientData):
    try:
        # Convert input data to DataFrame
        X = pd.DataFrame([data.model_dump()])
        
        # Engineer features
        X_processed = feature_engineer.engineer_features(X)
        
        # Make prediction
        prediction = model_trainer.predict(X_processed)
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(model_trainer.predict_proba(X_processed)[0][1])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain(data: PatientData):
    try:
        # Convert input data to DataFrame
        X = pd.DataFrame([data.model_dump()])
        
        # Engineer features
        X_processed = feature_engineer.engineer_features(X)
        
        # Generate explanation
        explanation = model_explainer.generate_explanation(X_processed)
        
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Engineer features
        X_processed = feature_engineer.engineer_features(df)
        
        # Make predictions
        predictions = model_trainer.predict(X_processed)
        probabilities = model_trainer.predict_proba(X_processed)
        
        # Prepare results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "id": i,
                "prediction": int(pred),
                "probability": float(prob[1])
            })
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    return {
        "model_type": model_trainer.model_type,
        "feature_importance": model_trainer.get_feature_importance().to_dict('records'),
        "model_metrics": model_trainer.get_model_metrics()
    } 