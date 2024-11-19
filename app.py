"""
FastAPI application for bus journey duration prediction.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
import numpy as np
import pandas as pd
from model_utils import load_model_artifacts, analyze_feature_importance_shap

# Initialize FastAPI app
app = FastAPI(
    title="Bus Journey Duration Predictor",
    description="API for predicting bus journey durations using machine learning",
    version="1.0.0"
)

# Load model artifacts at startup
try:
    artifacts = load_model_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_names = artifacts['feature_names']
    metadata = artifacts['metadata']
except Exception as e:
    print(f"Error loading model artifacts: {str(e)}")
    raise

class PredictionRequest(BaseModel):
    """
    Request model for prediction endpoint.
    Features should be a dictionary of feature names and their values.
    """
    features: Dict[str, float] = Field(..., description="Dictionary of feature names and values")

class PredictionResponse(BaseModel):
    """
    Response model for prediction endpoint.
    """
    prediction: float
    feature_importance: Optional[Dict[str, float]] = None

@app.get("/model-info")
async def get_model_info():
    """Get model metadata and information."""
    return {
        "model_type": metadata['model_type'],
        "n_features": metadata['n_features'],
        "feature_names": feature_names,
        "model_version": metadata.get('version', '1.0.0'),
        "training_date": metadata.get('training_date', 'Not specified'),
        "model_metrics": metadata.get('metrics', {})
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction for bus journey duration.
    
    Args:
        request: PredictionRequest object containing feature values
        
    Returns:
        PredictionResponse object containing prediction and feature importance
    """
    try:
        # Validate features
        if not request.features:
            raise HTTPException(status_code=400, detail="No features provided")
        
        # Check if all required features are present
        missing_features = set(feature_names) - set(request.features.keys())
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {', '.join(missing_features)}"
            )
        
        # Check for invalid features
        invalid_features = set(request.features.keys()) - set(feature_names)
        if invalid_features:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid features provided: {', '.join(invalid_features)}"
            )
        
        # Create feature array in correct order
        X = pd.DataFrame([request.features])[feature_names]
        
        # Scale features if scaler exists
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Make prediction
        prediction = float(model.predict(X_scaled)[0])
        
        # Calculate feature importance using SHAP values
        try:
            feature_importance = analyze_feature_importance_shap(
                model, X_scaled, feature_names
            )
        except Exception as e:
            print(f"Warning: Could not calculate feature importance: {str(e)}")
            feature_importance = None
        
        return PredictionResponse(
            prediction=prediction,
            feature_importance=feature_importance
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}
