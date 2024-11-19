"""
Utility functions for model training, evaluation, and feature engineering.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import learning_curve
import shap
from typing import List, Tuple, Dict, Any

def dataframe_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a detailed summary of a DataFrame's columns.
    """
    report = pd.DataFrame(columns=['Column', 'Data Type', 'Unique Count', 'Unique Sample', 'Missing Values', 'Missing Percentage'])
    for column in df.columns:
        data_type = df[column].dtype
        unique_count = df[column].nunique()
        unique_sample = df[column].unique()[:5]
        missing_values = df[column].isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        report = pd.concat([report, pd.DataFrame({
            'Column': [column],
            'Data Type': [data_type],
            'Unique Count': [unique_count],
            'Unique Sample': [unique_sample],
            'Missing Values': [missing_values],
            'Missing Percentage': [missing_percentage.round(4)]
        })], ignore_index=True)
    return report

def create_cyclical_features(df: pd.DataFrame, col: str, period: int) -> Tuple[pd.Series, pd.Series]:
    """Create sine and cosine features from a cyclical numerical column."""
    sin_feature = np.sin(2 * np.pi * df[col] / period)
    cos_feature = np.cos(2 * np.pi * df[col] / period)
    return sin_feature, cos_feature

def evaluate_model(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                  y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model performance using multiple metrics."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    return metrics

def analyze_feature_importance_shap(model, X: pd.DataFrame, feature_names: List[str]) -> Dict[str, float]:
    """Calculate feature importance using SHAP values."""
    try:
        # Create SHAP explainer
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Calculate mean absolute SHAP values for each feature
        feature_importance = {}
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        for idx, importance in enumerate(mean_abs_shap):
            feature_importance[feature_names[idx]] = float(importance)
        
        # Normalize to sum to 1
        total_importance = sum(feature_importance.values())
        feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        print(f"Warning: Could not calculate SHAP values: {str(e)}")
        return {}

def save_model_artifacts(model: Any, feature_names: List[str], 
                        scaler: Any = None, output_dir: str = 'model_artifacts') -> None:
    """Save model and associated artifacts for deployment."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    joblib.dump(model, os.path.join(output_dir, 'model.joblib'))
    
    # Save scaler if provided
    if scaler is not None:
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    
    # Save feature names
    joblib.dump(feature_names, os.path.join(output_dir, 'feature_names.joblib'))
    
    # Save metadata
    metadata = {
        'model_type': type(model).__name__,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'has_scaler': scaler is not None
    }
    joblib.dump(metadata, os.path.join(output_dir, 'metadata.joblib'))

def load_model_artifacts(input_dir: str = 'model_artifacts') -> Dict[str, Any]:
    """Load model and associated artifacts."""
    try:
        # Load model
        model = joblib.load(os.path.join(input_dir, 'model.joblib'))
        
        # Load feature names
        feature_names = joblib.load(os.path.join(input_dir, 'feature_names.joblib'))
        
        # Load metadata
        metadata = joblib.load(os.path.join(input_dir, 'metadata.joblib'))
        
        # Load scaler if it exists
        scaler_path = os.path.join(input_dir, 'scaler.joblib')
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        return {
            'model': model,
            'feature_names': feature_names,
            'metadata': metadata,
            'scaler': scaler
        }
    except Exception as e:
        raise RuntimeError(f"Error loading model artifacts: {str(e)}")
