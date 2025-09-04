#!/usr/bin/env python3
"""
Example usage of the Anomaly Detection API
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# API base URL
BASE_URL = "http://localhost:8000"

def create_sample_data():
    """Create sample time series data with anomalies."""
    np.random.seed(42)
    
    # Generate normal data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_days = len(dates)
    
    # Normal price data (trending upward with some noise)
    base_price = 100
    trend = np.linspace(0, 20, n_days)
    noise = np.random.normal(0, 2, n_days)
    normal_prices = base_price + trend + noise
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_days, size=20, replace=False)
    anomaly_prices = normal_prices.copy()
    
    for idx in anomaly_indices:
        # Add large price spikes or drops
        if np.random.random() > 0.5:
            anomaly_prices[idx] += np.random.normal(0, 15)  # Large spike
        else:
            anomaly_prices[idx] -= np.random.normal(0, 15)  # Large drop
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'price': anomaly_prices,
        'volume': np.random.normal(1000, 200, n_days),
        'instrument_id': 'AAPL'
    })
    
    return df

def upload_dataset(df):
    """Upload dataset to the API."""
    print("📤 Uploading dataset...")
    
    # Save to CSV
    csv_data = df.to_csv(index=False)
    
    files = {'file': ('sample_data.csv', csv_data, 'text/csv')}
    response = requests.post(f"{BASE_URL}/upload-dataset", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Dataset uploaded successfully!")
        print(f"   Dataset ID: {result['dataset_id']}")
        print(f"   Rows: {result['data_info']['rows']}")
        print(f"   Columns: {result['data_info']['columns']}")
        return result['dataset_id']
    else:
        print(f"❌ Error uploading dataset: {response.text}")
        return None

def train_model(dataset_id):
    """Train an anomaly detection model."""
    print("\n🤖 Training model...")
    
    training_request = {
        "algorithm": "isolation_forest",
        "contamination": 0.1,
        "target_column": "price",
        "feature_columns": ["volume"]  # Optional: specify features
    }
    
    response = requests.post(
        f"{BASE_URL}/train/{dataset_id}",
        json=training_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Model trained successfully!")
        print(f"   Model ID: {result['model_id']}")
        print(f"   Anomalies detected: {result['training_stats']['anomalies_detected']}")
        print(f"   Anomaly rate: {result['training_stats']['anomaly_rate']:.2%}")
        return result['model_id']
    else:
        print(f"❌ Error training model: {response.text}")
        return None

def test_single_prediction(model_id):
    """Test single prediction."""
    print("\n🔍 Testing single prediction...")
    
    # Test with normal data
    normal_data = {
        "price": 120.5,
        "volume": 950
    }
    
    prediction_request = {
        "model_id": model_id,
        "data": normal_data
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=prediction_request)
    
    if response.status_code == 200:
        result = response.json()
        prediction = result['prediction']
        print(f"✅ Normal data prediction:")
        print(f"   Is anomaly: {prediction['is_anomaly']}")
        print(f"   Score: {prediction['score']:.4f}")
        print(f"   Confidence: {prediction['confidence']}")
    
    # Test with anomalous data
    anomaly_data = {
        "price": 200.0,  # Much higher than normal
        "volume": 950
    }
    
    prediction_request["data"] = anomaly_data
    response = requests.post(f"{BASE_URL}/predict", json=prediction_request)
    
    if response.status_code == 200:
        result = response.json()
        prediction = result['prediction']
        print(f"✅ Anomalous data prediction:")
        print(f"   Is anomaly: {prediction['is_anomaly']}")
        print(f"   Score: {prediction['score']:.4f}")
        print(f"   Confidence: {prediction['confidence']}")

def test_batch_prediction(model_id, df):
    """Test batch prediction."""
    print("\n📊 Testing batch prediction...")
    
    # Take a sample of the data
    sample_data = df.sample(10).to_dict('records')
    
    batch_request = {
        "model_id": model_id,
        "data": sample_data
    }
    
    response = requests.post(f"{BASE_URL}/predict-batch", json=batch_request)
    
    if response.status_code == 200:
        result = response.json()
        predictions = result['predictions']
        print(f"✅ Batch prediction completed!")
        print(f"   Total predictions: {len(predictions['predictions'])}")
        print(f"   Anomalies found: {predictions['anomaly_count']}")
        print(f"   Anomaly rate: {predictions['anomaly_rate']:.2%}")
        
        # Show individual predictions
        for i, (is_anomaly, score) in enumerate(zip(predictions['predictions'], predictions['scores'])):
            print(f"   Sample {i+1}: Anomaly={is_anomaly}, Score={score:.4f}")

def list_models():
    """List all trained models."""
    print("\n📋 Listing models...")
    
    response = requests.get(f"{BASE_URL}/models")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Found {result['count']} models:")
        for model in result['models']:
            print(f"   - {model['model_id']} ({model['algorithm']})")
            print(f"     Target: {model['target_column']}")
            print(f"     Created: {model['created_at']}")
    else:
        print(f"❌ Error listing models: {response.text}")

def main():
    """Main example workflow."""
    print("🚀 Anomaly Detection API Example")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ API is not running. Please start it with: uvicorn main:app --reload")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please start it with: uvicorn main:app --reload")
        return
    
    print("✅ API is running!")
    
    # Create sample data
    print("\n📊 Creating sample data...")
    df = create_sample_data()
    print(f"✅ Created {len(df)} records with anomalies")
    
    # Upload dataset
    dataset_id = upload_dataset(df)
    if not dataset_id:
        return
    
    # Train model
    model_id = train_model(dataset_id)
    if not model_id:
        return
    
    # Test predictions
    test_single_prediction(model_id)
    test_batch_prediction(model_id, df)
    
    # List models
    list_models()
    
    print("\n🎉 Example completed successfully!")
    print(f"💡 You can now use model '{model_id}' for predictions")

if __name__ == "__main__":
    main()
