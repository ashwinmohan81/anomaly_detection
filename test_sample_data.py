#!/usr/bin/env python3
"""
Test anomaly detection on the generated sample stock data
"""

import requests
import pandas as pd
import time
import json

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def wait_for_api():
    """Wait for API to be ready."""
    print("⏳ Waiting for API to start...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
        print(f"   Attempt {i+1}/30...")
    
    print("❌ API failed to start")
    return False

def upload_sample_data():
    """Upload the sample stock data."""
    print("\n📤 Uploading sample stock data...")
    
    try:
        # Read the sample data
        df = pd.read_csv("sample_stock_data.csv")
        print(f"   Loaded {len(df)} records from sample_stock_data.csv")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        # Convert to CSV string
        csv_data = df.to_csv(index=False)
        
        # Upload to API
        files = {'file': ('sample_stock_data.csv', csv_data, 'text/csv')}
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
            
    except Exception as e:
        print(f"❌ Error reading sample data: {e}")
        return None

def train_models(dataset_id):
    """Train multiple models on the sample data."""
    print("\n🤖 Training models...")
    
    models = {}
    
    # Test different algorithms
    algorithms = [
        ("isolation_forest", "Isolation Forest"),
        ("one_class_svm", "One-Class SVM"),
        ("local_outlier_factor", "Local Outlier Factor")
    ]
    
    for algorithm, name in algorithms:
        print(f"\n   Training {name}...")
        
        training_request = {
            "algorithm": algorithm,
            "contamination": 0.1,  # Expect 10% anomalies
            "target_column": "close",  # Detect anomalies in closing price
            "feature_columns": ["open", "high", "low", "volume"]  # Use other OHLC + volume as features
        }
        
        response = requests.post(
            f"{BASE_URL}/train/{dataset_id}",
            json=training_request
        )
        
        if response.status_code == 200:
            result = response.json()
            model_id = result['model_id']
            stats = result['training_stats']
            
            print(f"   ✅ {name} trained successfully!")
            print(f"      Model ID: {model_id}")
            print(f"      Anomalies detected: {stats['anomalies_detected']}")
            print(f"      Anomaly rate: {stats['anomaly_rate']:.2%}")
            print(f"      Features used: {len(stats['feature_columns'])}")
            
            models[algorithm] = {
                'model_id': model_id,
                'name': name,
                'stats': stats
            }
        else:
            print(f"   ❌ Error training {name}: {response.text}")
    
    return models

def test_single_predictions(models):
    """Test single predictions with different data points."""
    print("\n🔍 Testing single predictions...")
    
    # Test data points
    test_cases = [
        {"name": "Normal price", "data": {"open": 150.0, "high": 151.0, "low": 149.0, "close": 150.5, "volume": 50000000}},
        {"name": "High price spike", "data": {"open": 150.0, "high": 200.0, "low": 149.0, "close": 195.0, "volume": 50000000}},
        {"name": "Low price drop", "data": {"open": 150.0, "high": 151.0, "low": 100.0, "close": 105.0, "volume": 50000000}},
        {"name": "Volume spike", "data": {"open": 150.0, "high": 151.0, "low": 149.0, "close": 150.5, "volume": 500000000}},
    ]
    
    for test_case in test_cases:
        print(f"\n   Testing: {test_case['name']}")
        print(f"   Data: {test_case['data']}")
        
        for algorithm, model_info in models.items():
            prediction_request = {
                "model_id": model_info['model_id'],
                "data": test_case['data']
            }
            
            response = requests.post(f"{BASE_URL}/predict", json=prediction_request)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                print(f"   {model_info['name']}: {'🚨 ANOMALY' if prediction['is_anomaly'] else '✅ Normal'} "
                      f"(score: {prediction['score']:.4f}, confidence: {prediction['confidence']})")
            else:
                print(f"   {model_info['name']}: ❌ Error - {response.text}")

def test_batch_predictions(models, dataset_id):
    """Test batch predictions on sample data."""
    print("\n📊 Testing batch predictions...")
    
    # Get a sample of the original data
    df = pd.read_csv("sample_stock_data.csv")
    sample_data = df.sample(20).to_dict('records')
    
    print(f"   Testing on {len(sample_data)} sample records...")
    
    for algorithm, model_info in models.items():
        print(f"\n   {model_info['name']} batch results:")
        
        batch_request = {
            "model_id": model_info['model_id'],
            "data": sample_data
        }
        
        response = requests.post(f"{BASE_URL}/predict-batch", json=batch_request)
        
        if response.status_code == 200:
            result = response.json()
            predictions = result['predictions']
            print(f"      Total predictions: {len(predictions['predictions'])}")
            print(f"      Anomalies found: {predictions['anomaly_count']}")
            print(f"      Anomaly rate: {predictions['anomaly_rate']:.2%}")
            
            # Show some individual predictions
            for i, (is_anomaly, score) in enumerate(zip(predictions['predictions'][:5], predictions['scores'][:5])):
                status = "🚨" if is_anomaly else "✅"
                print(f"      Sample {i+1}: {status} Score={score:.4f}")
        else:
            print(f"      ❌ Error: {response.text}")

def analyze_results(models):
    """Analyze and compare model results."""
    print("\n📈 Model Comparison Summary:")
    print("=" * 60)
    
    for algorithm, model_info in models.items():
        stats = model_info['stats']
        print(f"\n{model_info['name']}:")
        print(f"   Algorithm: {stats['algorithm']}")
        print(f"   Contamination: {stats['contamination']}")
        print(f"   Anomalies detected: {stats['anomalies_detected']}")
        print(f"   Anomaly rate: {stats['anomaly_rate']:.2%}")
        print(f"   Features used: {len(stats['feature_columns'])}")
        print(f"   Model ID: {model_info['model_id']}")

def main():
    """Main test workflow."""
    print("🚀 Testing Anomaly Detection on Sample Stock Data")
    print("=" * 60)
    
    # Wait for API
    if not wait_for_api():
        return
    
    # Upload sample data
    dataset_id = upload_sample_data()
    if not dataset_id:
        return
    
    # Train models
    models = train_models(dataset_id)
    if not models:
        print("❌ No models were trained successfully")
        return
    
    # Test predictions
    test_single_predictions(models)
    test_batch_predictions(models, dataset_id)
    
    # Analyze results
    analyze_results(models)
    
    print("\n🎉 Testing completed successfully!")
    print(f"💡 You can now use these models for anomaly detection:")
    for algorithm, model_info in models.items():
        print(f"   - {model_info['name']}: {model_info['model_id']}")

if __name__ == "__main__":
    main()
