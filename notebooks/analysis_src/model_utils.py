"""
Utility functions for model training, evaluation, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from typing import List, Tuple, Dict, Any

def dataframe_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a detailed summary of a DataFrame's columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame containing column information including data types, unique values, and missing data
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
    """
    Create sine and cosine features from a cyclical numerical column.
    
    Args:
        df: Input DataFrame
        col: Name of the column to transform
        period: The period of the cycle (e.g., 24 for hours, 7 for days of week)
        
    Returns:
        Tuple of (sine feature, cosine feature)
    """
    sin_feature = np.sin(2 * np.pi * df[col] / period)
    cos_feature = np.cos(2 * np.pi * df[col] / period)
    return sin_feature, cos_feature

def evaluate_model(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                  y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate a model's performance using multiple metrics.
    
    Args:
        model: Trained model object
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae_train': mean_absolute_error(y_train, y_train_pred),
        'mae_test': mean_absolute_error(y_test, y_test_pred),
        'r2_train': r2_score(y_train, y_train_pred),
        'r2_test': r2_score(y_test, y_test_pred)
    }
    
    # Add percentage differences
    metrics['rmse_diff_pct'] = ((metrics['rmse_test'] - metrics['rmse_train']) / metrics['rmse_train']) * 100
    metrics['mae_diff_pct'] = ((metrics['mae_test'] - metrics['mae_train']) / metrics['mae_train']) * 100
    metrics['r2_diff_pct'] = ((metrics['r2_test'] - metrics['r2_train']) / metrics['r2_train']) * 100
    
    return metrics

def select_features(X: pd.DataFrame, y: pd.Series, method: str = 'f_regression', 
                   k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select top k features using various feature selection methods.
    
    Args:
        X: Feature DataFrame
        y: Target variable
        method: Feature selection method ('f_regression' or 'mutual_info')
        k: Number of features to select
        
    Returns:
        Tuple of (selected features DataFrame, list of selected feature names)
    """
    if method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    return pd.DataFrame(X_selected, columns=selected_features), selected_features

def plot_feature_importance(model: Any, feature_names: List[str], top_n: int = 20) -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for plotting
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort and get top features
    feat_imp = feat_imp.sort_values('importance', ascending=False).head(top_n)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp, x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def plot_learning_curves(model, X_train, y_train, cv=5):
    """
    Plot learning curves to analyze model performance with varying training data size.
    
    Args:
        model: Trained model object
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
    """
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_root_mean_squared_error'
    )
    
    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = -np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training Examples')
    plt.ylabel('RMSE')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def tune_hyperparameters(model_class, param_grid, X_train, y_train, cv=5):
    """
    Perform grid search cross-validation for hyperparameter tuning.
    
    Args:
        model_class: Untrained model class (e.g., RandomForestRegressor)
        param_grid: Dictionary of parameters to tune
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        
    Returns:
        Tuple of (best model, dictionary of results)
    """
    from sklearn.model_selection import GridSearchCV
    
    grid_search = GridSearchCV(
        model_class(),
        param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    results = {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_,  # Convert back to RMSE
        'cv_results': pd.DataFrame(grid_search.cv_results_)
    }
    
    return grid_search.best_estimator_, results

def plot_residuals(y_true, y_pred):
    """
    Create diagnostic plots for model residuals.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Residual Analysis Plots')
    
    # Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    
    # Residual Distribution
    axes[0, 1].hist(residuals, bins=30, edgecolor='black')
    axes[0, 1].set_xlabel('Residual Value')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Residual Distribution')
    
    # Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    # Residual Scatter
    axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residual Scatter')
    
    plt.tight_layout()
    plt.show()

def analyze_predictions(model, X_test, y_test, feature_names=None):
    """
    Comprehensive analysis of model predictions.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True test values
        feature_names: List of feature names
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_absolute_percentage_error
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)
    
    print("Model Performance Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")
    
    # Create prediction vs actual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual')
    plt.tight_layout()
    plt.show()
    
    # Plot residuals
    plot_residuals(y_test, y_pred)
    
    # Feature importance for tree-based models
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        plot_feature_importance(model, feature_names)

def create_error_analysis(y_true, y_pred, feature_df):
    """
    Analyze prediction errors in relation to features.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        feature_df: DataFrame of features used for prediction
        
    Returns:
        DataFrame with error analysis
    """
    errors = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'error': y_true - y_pred,
        'abs_error': np.abs(y_true - y_pred),
        'pct_error': np.abs((y_true - y_pred) / y_true) * 100
    })
    
    # Combine with features
    error_analysis = pd.concat([errors, feature_df], axis=1)
    
    # Calculate error statistics by feature quantiles
    error_stats = {}
    numeric_features = feature_df.select_dtypes(include=[np.number]).columns
    
    for feature in numeric_features:
        error_stats[feature] = error_analysis.groupby(
            pd.qcut(error_analysis[feature], q=5)
        )['abs_error'].agg(['mean', 'std', 'count'])
    
    return error_analysis, error_stats

def create_interaction_features(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]], 
                              operations: List[str] = ['multiply', 'divide', 'add', 'subtract']) -> pd.DataFrame:
    """
    Create interaction features between pairs of numerical columns.
    
    Args:
        df: Input DataFrame
        feature_pairs: List of tuples containing feature pairs to interact
        operations: List of operations to perform ('multiply', 'divide', 'add', 'subtract')
        
    Returns:
        DataFrame with interaction features added
    """
    result = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 not in df.columns or feat2 not in df.columns:
            continue
            
        if 'multiply' in operations:
            result[f'{feat1}_times_{feat2}'] = df[feat1] * df[feat2]
        if 'divide' in operations and (df[feat2] != 0).all():
            result[f'{feat1}_div_{feat2}'] = df[feat1] / df[feat2]
        if 'add' in operations:
            result[f'{feat1}_plus_{feat2}'] = df[feat1] + df[feat2]
        if 'subtract' in operations:
            result[f'{feat1}_minus_{feat2}'] = df[feat1] - df[feat2]
    
    return result

def create_polynomial_features(df: pd.DataFrame, features: List[str], degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial features for specified columns.
    
    Args:
        df: Input DataFrame
        features: List of features to create polynomials for
        degree: Maximum polynomial degree
        
    Returns:
        DataFrame with polynomial features added
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    result = df.copy()
    
    for feature in features:
        if feature not in df.columns:
            continue
            
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        feature_poly = poly.fit_transform(df[[feature]])
        
        # Add polynomial features
        for i in range(2, degree + 1):
            result[f'{feature}_power_{i}'] = feature_poly[:, i]
    
    return result

def create_lag_features(df: pd.DataFrame, features: List[str], 
                       time_column: str, group_columns: List[str] = None,
                       lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
    """
    Create lagged features for time series data.
    
    Args:
        df: Input DataFrame
        features: List of features to create lags for
        time_column: Column containing time information
        group_columns: List of columns to group by (e.g., route_id)
        lags: List of lag values to create
        
    Returns:
        DataFrame with lagged features added
    """
    result = df.copy()
    result = result.sort_values(time_column)
    
    for feature in features:
        if feature not in df.columns:
            continue
            
        if group_columns:
            for lag in lags:
                result[f'{feature}_lag_{lag}'] = result.groupby(group_columns)[feature].shift(lag)
        else:
            for lag in lags:
                result[f'{feature}_lag_{lag}'] = result[feature].shift(lag)
    
    return result

def analyze_feature_importance_shap(model, X: pd.DataFrame, max_display: int = 20):
    """
    Analyze feature importance using SHAP values.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        max_display: Maximum number of features to display
    """
    import shap
    import matplotlib.pyplot as plt
    
    # Create SHAP explainer
    if hasattr(model, 'predict_proba'):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # If regression, shap_values is already a numpy array
    # If classification, it's a list of arrays, take the first class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Plot SHAP summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # Plot SHAP dependence plots for top features
    feature_importance = np.abs(shap_values).mean(0)
    top_features = X.columns[np.argsort(-feature_importance)[:5]]
    
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X, show=False)
        plt.title(f'SHAP Dependence Plot: {feature}')
        plt.tight_layout()
        plt.show()

def create_stacking_ensemble(base_models: Dict[str, Any], meta_model: Any,
                           X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, cv: int = 5) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
    """
    Create a stacking ensemble model.
    
    Args:
        base_models: Dictionary of base models
        meta_model: Meta-learner model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        cv: Number of cross-validation folds
        
    Returns:
        Tuple of (trained meta-model, train meta-features, test meta-features)
    """
    from sklearn.model_selection import KFold
    
    # Initialize
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    meta_train = np.zeros((X_train.shape[0], len(base_models)))
    meta_test = np.zeros((X_test.shape[0], len(base_models)))
    
    # Generate meta-features
    for i, (name, model) in enumerate(base_models.items()):
        # Create test meta-features
        model.fit(X_train, y_train)
        meta_test[:, i] = model.predict(X_test)
        
        # Create train meta-features
        for train_idx, val_idx in kf.split(X_train):
            # Get train and validation sets
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            
            # Train model and make predictions
            model.fit(X_train_fold, y_train_fold)
            meta_train[val_idx, i] = model.predict(X_val_fold)
    
    # Prepare meta-features
    meta_train = pd.DataFrame(meta_train, columns=base_models.keys())
    meta_test = pd.DataFrame(meta_test, columns=base_models.keys())
    
    # Train meta-model
    meta_model.fit(meta_train, y_train)
    
    return meta_model, meta_train, meta_test

def save_model_artifacts(model: Any, feature_names: List[str], 
                        scaler: Any = None, output_dir: str = 'model_artifacts'):
    """
    Save model and associated artifacts for deployment.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        scaler: Fitted scaler object (optional)
        output_dir: Directory to save artifacts
    """
    import joblib
    import os
    import json
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    joblib.dump(model, os.path.join(output_dir, 'model.joblib'))
    
    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    
    # Save scaler if provided
    if scaler is not None:
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    
    # Create a simple metadata file
    metadata = {
        'model_type': type(model).__name__,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'has_scaler': scaler is not None,
        'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def load_model_artifacts(input_dir: str = 'model_artifacts') -> Dict[str, Any]:
    """
    Load model and associated artifacts.
    
    Args:
        input_dir: Directory containing model artifacts
        
    Returns:
        Dictionary containing model and artifacts
    """
    import joblib
    import json
    import os
    
    artifacts = {}
    
    # Load model
    artifacts['model'] = joblib.load(os.path.join(input_dir, 'model.joblib'))
    
    # Load feature names
    with open(os.path.join(input_dir, 'feature_names.json'), 'r') as f:
        artifacts['feature_names'] = json.load(f)
    
    # Load scaler if it exists
    scaler_path = os.path.join(input_dir, 'scaler.joblib')
    if os.path.exists(scaler_path):
        artifacts['scaler'] = joblib.load(scaler_path)
    
    # Load metadata
    with open(os.path.join(input_dir, 'metadata.json'), 'r') as f:
        artifacts['metadata'] = json.load(f)
    
    return artifacts

def create_prediction_api(model_dir: str = 'model_artifacts') -> str:
    """
    Create a FastAPI application for model deployment.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        String containing the FastAPI application code
    """
    api_code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict
import joblib
import json
import os

app = FastAPI(title="Bus Duration Prediction API")

# Load model artifacts
model_dir = "{model_dir}"
artifacts = {{}}

@app.on_event("startup")
async def load_artifacts():
    global artifacts
    artifacts["model"] = joblib.load(os.path.join(model_dir, "model.joblib"))
    
    with open(os.path.join(model_dir, "feature_names.json"), "r") as f:
        artifacts["feature_names"] = json.load(f)
    
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        artifacts["scaler"] = joblib.load(scaler_path)

class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: float
    feature_importance: Dict[str, float] = None

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Create feature DataFrame
        features = pd.DataFrame([request.features])
        
        # Ensure all required features are present
        missing_features = set(artifacts["feature_names"]) - set(features.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {{missing_features}}"
            )
        
        # Reorder features to match training data
        features = features[artifacts["feature_names"]]
        
        # Scale features if scaler exists
        if "scaler" in artifacts:
            features = pd.DataFrame(
                artifacts["scaler"].transform(features),
                columns=features.columns
            )
        
        # Make prediction
        prediction = float(artifacts["model"].predict(features)[0])
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(artifacts["model"], "feature_importances_"):
            feature_importance = dict(zip(
                artifacts["feature_names"],
                artifacts["model"].feature_importances_
            ))
        
        return PredictionResponse(
            prediction=prediction,
            feature_importance=feature_importance
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    with open(os.path.join(model_dir, "metadata.json"), "r") as f:
        return json.load(f)
'''.format(model_dir=model_dir)
    
    return api_code
