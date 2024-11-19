"""
Test script for validating model deployment and API functionality.
"""

import requests
import pandas as pd
import numpy as np
from time import sleep
import subprocess
import sys
import json
from typing import Dict, Any

def test_model_artifacts():
    """Test model artifact loading and basic functionality."""
    from model_utils import load_model_artifacts
    
    print("\nTesting model artifacts...")
    try:
        artifacts = load_model_artifacts()
        print("✓ Successfully loaded model artifacts")
        print(f"  - Model type: {artifacts['metadata']['model_type']}")
        print(f"  - Number of features: {artifacts['metadata']['n_features']}")
        return True
    except Exception as e:
        print(f"✗ Failed to load model artifacts: {str(e)}")
        return False

def start_api_server():
    """Start the FastAPI server using uvicorn."""
    print("\nStarting API server...")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--reload"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        sleep(5)  # Wait for server to start
        return process
    except Exception as e:
        print(f"✗ Failed to start API server: {str(e)}")
        return None

def test_api_endpoints(base_url: str = "http://localhost:8000"):
    """Test all API endpoints."""
    print("\nTesting API endpoints...")
    
    # Test model info endpoint
    try:
        response = requests.get(f"{base_url}/model-info")
        assert response.status_code == 200
        print("✓ Successfully retrieved model info")
        print(f"  Model metadata: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"✗ Failed to get model info: {str(e)}")
        return False
    
    # Test prediction endpoint with sample data
    try:
        # Load artifacts to get feature names
        from model_utils import load_model_artifacts
        artifacts = load_model_artifacts()
        feature_names = artifacts['feature_names']
        
        # Create sample features
        sample_features = {
            feature: np.random.rand() for feature in feature_names
        }
        
        # Make prediction request
        response = requests.post(
            f"{base_url}/predict",
            json={"features": sample_features}
        )
        
        assert response.status_code == 200
        result = response.json()
        print("✓ Successfully made prediction")
        print(f"  Prediction: {result['prediction']:.2f}")
        if result.get('feature_importance'):
            print("  Feature importance available")
        
        return True
    except Exception as e:
        print(f"✗ Failed to make prediction: {str(e)}")
        return False

def test_error_handling(base_url: str = "http://localhost:8000"):
    """Test API error handling."""
    print("\nTesting error handling...")
    
    # Test missing features
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"features": {}}
        )
        assert response.status_code == 400
        print("✓ Successfully handled missing features")
    except Exception as e:
        print(f"✗ Failed to handle missing features: {str(e)}")
        return False
    
    # Test invalid feature names
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"features": {"invalid_feature": 1.0}}
        )
        assert response.status_code == 400
        print("✓ Successfully handled invalid features")
    except Exception as e:
        print(f"✗ Failed to handle invalid features: {str(e)}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Starting model deployment tests...")
    
    # Test model artifacts
    if not test_model_artifacts():
        print("✗ Model artifacts test failed. Stopping tests.")
        return
    
    # Start API server
    server_process = start_api_server()
    if server_process is None:
        print("✗ Failed to start API server. Stopping tests.")
        return
    
    try:
        # Wait for server to start
        print("Waiting for server to start...")
        sleep(5)
        
        # Run API tests
        api_success = test_api_endpoints()
        error_handling_success = test_error_handling()
        
        if api_success and error_handling_success:
            print("\n✓ All tests passed successfully!")
        else:
            print("\n✗ Some tests failed. Check the logs above.")
    
    finally:
        # Clean up
        print("\nStopping API server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()
