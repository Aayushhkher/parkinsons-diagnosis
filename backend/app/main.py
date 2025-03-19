from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import logging
from datetime import datetime

# Import ML modules
from ml.features.engineer import FeatureEngineer
from ml.models.train import ModelTrainer
from ml.explainability.explainer import ModelExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Parkinson's Disease AI CDSS",
    description="AI-driven Clinical Decision Support System for Parkinson's Disease diagnosis and management",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and feature engineer
MODEL_PATH = Path("ml/models/saved")
feature_engineer = FeatureEngineer()
models = {}

def load_models():
    """Load trained models from disk."""
    try:
        models['svm'] = joblib.load(MODEL_PATH / "svm_model.joblib")
        models['rf'] = joblib.load(MODEL_PATH / "rf_model.joblib")
        models['nn'] = tf.keras.models.load_model(MODEL_PATH / "nn_model.h5")
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading models")

# Load models on startup
@app.on_event("startup")
async def startup_event():
    load_models()

class PatientData(BaseModel):
    """Schema for patient data input."""
    clinical_data: Dict[str, float]
    imaging_data: Optional[Dict[str, Any]] = None
    voice_data: Optional[Dict[str, Any]] = None
    motion_data: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    prediction: int
    probability: float
    model_used: str
    timestamp: str

class ExplanationResponse(BaseModel):
    """Schema for explanation response."""
    shap_values: Dict[str, float]
    lime_explanation: Dict[str, float]
    feature_importance: List[Dict[str, float]]

@app.post("/predict", response_model=PredictionResponse)
async def predict_pd(data: PatientData):
    """
    Make a prediction for Parkinson's Disease diagnosis.
    
    Args:
        data: Patient data including clinical, imaging, voice, and motion data
        
    Returns:
        Prediction and probability
    """
    try:
        # Extract features
        features = feature_engineer.extract_medical_features(data.dict())
        
        # Make predictions with different models
        predictions = {}
        probabilities = {}
        
        for model_name, model in models.items():
            if model_name == 'nn':
                pred = (model.predict(features) > 0.5).astype(int)[0][0]
                prob = float(model.predict(features)[0][0])
            else:
                pred = model.predict(features)[0]
                prob = float(model.predict_proba(features)[0][1])
                
            predictions[model_name] = pred
            probabilities[model_name] = prob
            
        # Use ensemble prediction (majority voting)
        final_pred = int(np.mean(list(predictions.values())) > 0.5)
        final_prob = np.mean(list(probabilities.values()))
        
        return PredictionResponse(
            prediction=final_pred,
            probability=final_prob,
            model_used="ensemble",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(data: PatientData):
    """
    Generate explanations for a prediction.
    
    Args:
        data: Patient data including clinical, imaging, voice, and motion data
        
    Returns:
        SHAP and LIME explanations
    """
    try:
        # Extract features
        features = feature_engineer.extract_medical_features(data.dict())
        
        # Initialize explainer with ensemble model
        explainer = ModelExplainer(models['rf'], features.columns.tolist())
        
        # Get explanations
        shap_results = explainer.get_shap_values(features)
        lime_results = explainer.get_lime_explanation(features.iloc[0])
        
        return ExplanationResponse(
            shap_values=shap_results['feature_importance'].to_dict('records'),
            lime_explanation=lime_results['feature_importance'].to_dict('records'),
            feature_importance=shap_results['feature_importance'].to_dict('records')
        )
        
    except Exception as e:
        logger.error(f"Error generating explanations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-data")
async def upload_patient_data(file: UploadFile = File(...)):
    """
    Upload patient data file for batch processing.
    
    Args:
        file: CSV file containing patient data
        
    Returns:
        Processing status and results
    """
    try:
        # Read uploaded file
        df = pd.read_csv(file.file)
        
        # Process data
        features = feature_engineer.preprocess_data(df)
        
        # Make predictions
        predictions = {}
        for model_name, model in models.items():
            if model_name == 'nn':
                preds = (model.predict(features) > 0.5).astype(int).flatten()
            else:
                preds = model.predict(features)
            predictions[model_name] = preds.tolist()
            
        return {
            "status": "success",
            "predictions": predictions,
            "processed_rows": len(df)
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": list(models.keys())} 