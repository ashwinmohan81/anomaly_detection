#!/usr/bin/env python3
"""
Example of supervised anomaly detection with labeled data
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# API base URL
BASE_URL = "http://localhost:8000"

def create_labeled_sample_data():
    """Create sample data with known anomalies (labels)."""
    np.random.seed(42)
    
    # Generate normal data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_days = len(dates)
    
    # Normal price data
    base_price = 100
    trend = np.linspace(0, 20, n_days)
    noise = np.random.normal(0, 2, n_days)
    normal_prices = base_price + trend + noise
    
    # Create labels (0 = normal, 1 = anomaly)
    labels = np.zeros(n_days)
    
    # Add known anomalies with labels
    anomaly_indices = np.random.choice(n_days, size=30, replace=False)
    anomaly_prices = normal_prices.copy()
    
    for idx in anomaly_indices:
        # Add large price spikes or drops
        if np.random.random() > 0.5:
            anomaly_prices[idx] += np.random.normal(0, 15)  # Large spike
        else:
            anomaly_prices[idx] -= np.random.normal(0, 15)  # Large drop
        
        # Mark as anomaly
        labels[idx] = 1
    
    # Create DataFrame with labels
    df = pd.DataFrame({
        'date': dates,
        'price': anomaly_prices,
        'volume': np.random.normal(1000, 200, n_days),
        'instrument_id': 'AAPL',
        'is_anomaly': labels  # This is our label column
    })
    
    return df

def upload_labeled_dataset(df):
    """Upload labeled dataset to the API."""
    print("📤 Uploading labeled dataset...")
    
    # Save to CSV
    csv_data = df.to_csv(index=False)
    
    files = {'file': ('labeled_data.csv', csv_data, 'text/csv')}
    response = requests.post(f"{BASE_URL}/upload-dataset", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Labeled dataset uploaded successfully!")
        print(f"   Dataset ID: {result['dataset_id']}")
        print(f"   Rows: {result['data_info']['rows']}")
        print(f"   Columns: {result['data_info']['columns']}")
        print(f"   True anomalies: {df['is_anomaly'].sum()}")
        return result['dataset_id']
    else:
        print(f"❌ Error uploading dataset: {response.text}")
        return None

def train_supervised_model(dataset_id):
    """Train a supervised anomaly detection model."""
    print("\n🤖 Training supervised model...")
    
    training_request = {
        "algorithm": "random_forest",  # Use supervised algorithm
        "contamination": 0.1,
        "target_column": "price",
        "feature_columns": ["volume"],
        "label_column": "is_anomaly"  # Specify the label column
    }
    
    response = requests.post(
        f"{BASE_URL}/train/{dataset_id}",
        json=training_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Supervised model trained successfully!")
        print(f"   Model ID: {result['model_id']}")
        print(f"   F1-Score: {result['training_stats']['f1_score']:.3f}")
        print(f"   Precision: {result['training_stats']['precision']:.3f}")
        print(f"   Recall: {result['training_stats']['recall']:.3f}")
        print(f"   True anomalies: {result['training_stats']['true_anomalies']}")
        print(f"   Predicted anomalies: {result['training_stats']['predicted_anomalies']}")
        return result['model_id']
    else:
        print(f"❌ Error training model: {response.text}")
        return None

def train_unsupervised_model(dataset_id):
    """Train an unsupervised model for comparison."""
    print("\n🤖 Training unsupervised model for comparison...")
    
    training_request = {
        "algorithm": "isolation_forest",
        "contamination": 0.1,
        "target_column": "price",
        "feature_columns": ["volume"]
        # No label_column = unsupervised
    }
    
    response = requests.post(
        f"{BASE_URL}/train/{dataset_id}",
        json=training_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Unsupervised model trained successfully!")
        print(f"   Model ID: {result['model_id']}")
        print(f"   Anomalies detected: {result['training_stats']['anomalies_detected']}")
        print(f"   Anomaly rate: {result['training_stats']['anomaly_rate']:.2%}")
        return result['model_id']
    else:
        print(f"❌ Error training model: {response.text}")
        return None

def compare_models(supervised_model_id, unsupervised_model_id, df):
    """Compare supervised vs unsupervised model performance."""
    print("\n📊 Comparing model performance...")
    
    # Test on a sample of data
    test_data = df.sample(20).to_dict('records')
    
    # Test supervised model
    supervised_request = {
        "model_id": supervised_model_id,
        "data": test_data
    }
    
    response = requests.post(f"{BASE_URL}/predict-batch", json=supervised_request)
    if response.status_code == 200:
        supervised_result = response.json()['predictions']
        supervised_anomalies = supervised_result['anomaly_count']
        print(f"✅ Supervised model predictions: {supervised_anomalies} anomalies")
    
    # Test unsupervised model
    unsupervised_request = {
        "model_id": unsupervised_model_id,
        "data": test_data
    }
    
    response = requests.post(f"{BASE_URL}/predict-batch", json=unsupervised_request)
    if response.status_code == 200:
        unsupervised_result = response.json()['predictions']
        unsupervised_anomalies = unsupervised_result['anomaly_count']
        print(f"✅ Unsupervised model predictions: {unsupervised_anomalies} anomalies")
    
    # Get true labels for comparison
    true_anomalies = sum(record['is_anomaly'] for record in test_data)
    print(f"✅ True anomalies in test set: {true_anomalies}")
    
    print(f"\n📈 Performance Summary:")
    print(f"   True anomalies: {true_anomalies}")
    print(f"   Supervised detected: {supervised_anomalies}")
    print(f"   Unsupervised detected: {unsupervised_anomalies}")

def test_single_prediction(model_id, df):
    """Test single prediction with known anomaly."""
    print("\n🔍 Testing single prediction...")
    
    # Find a known anomaly
    anomaly_row = df[df['is_anomaly'] == 1].iloc[0]
    
    test_data = {
        "price": anomaly_row['price'],
        "volume": anomaly_row['volume']
    }
    
    prediction_request = {
        "model_id": model_id,
        "data": test_data
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=prediction_request)
    
    if response.status_code == 200:
        result = response.json()
        prediction = result['prediction']
        print(f"✅ Testing known anomaly:")
        print(f"   Price: {test_data['price']:.2f}")
        print(f"   Volume: {test_data['volume']:.2f}")
        print(f"   True label: Anomaly")
        print(f"   Predicted: {'Anomaly' if prediction['is_anomaly'] else 'Normal'}")
        print(f"   Score: {prediction['score']:.4f}")
        print(f"   Confidence: {prediction['confidence']}")

def main():
    """Main supervised learning example workflow."""
    print("🚀 Supervised Anomaly Detection Example")
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
    
    # Create labeled sample data
    print("\n📊 Creating labeled sample data...")
    df = create_labeled_sample_data()
    print(f"✅ Created {len(df)} records with {df['is_anomaly'].sum()} known anomalies")
    
    # Upload dataset
    dataset_id = upload_labeled_dataset(df)
    if not dataset_id:
        return
    
    # Train supervised model
    supervised_model_id = train_supervised_model(dataset_id)
    if not supervised_model_id:
        return
    
    # Train unsupervised model for comparison
    unsupervised_model_id = train_unsupervised_model(dataset_id)
    if not unsupervised_model_id:
        return
    
    # Compare models
    compare_models(supervised_model_id, unsupervised_model_id, df)
    
    # Test single prediction
    test_single_prediction(supervised_model_id, df)
    
    print("\n🎉 Supervised learning example completed!")
    print(f"💡 Supervised model '{supervised_model_id}' learned from labeled data")
    print(f"💡 Unsupervised model '{unsupervised_model_id}' for comparison")

if __name__ == "__main__":
    main()
