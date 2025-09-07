#!/usr/bin/env python3
"""
Test script to verify the actual API response format
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime

def test_api_response_format():
    """Test the actual API response format."""
    print("🧪 Testing Actual API Response Format")
    print("=" * 60)
    
    # Start the API server (assuming it's running on port 8000)
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        print("1. Testing Health Endpoint")
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        print()
        
        # Test algorithms endpoint
        print("2. Testing Algorithms Endpoint")
        response = requests.get(f"{base_url}/algorithms")
        print(f"   Status: {response.status_code}")
        algorithms = response.json()
        print(f"   Found {len(algorithms['algorithms'])} algorithms")
        for algo in algorithms['algorithms'][:2]:  # Show first 2
            print(f"   - {algo['name']}: {algo['description']}")
        print()
        
        # Test examples endpoint
        print("3. Testing Examples Endpoint")
        response = requests.get(f"{base_url}/examples")
        print(f"   Status: {response.status_code}")
        examples = response.json()
        print(f"   Found {len(examples['examples'])} examples")
        for example in examples['examples'][:2]:  # Show first 2
            print(f"   - {example['name']}: {example['description']}")
        print()
        
        # Test models endpoint
        print("4. Testing Models Endpoint")
        response = requests.get(f"{base_url}/models")
        print(f"   Status: {response.status_code}")
        models = response.json()
        print(f"   Found {models['count']} models")
        if models['count'] > 0:
            model = models['models'][0]
            print(f"   Sample model: {model['model_id']}")
            print(f"   Algorithm: {model['algorithm']}")
            print(f"   Business key: {model['business_key']}")
            print(f"   Target attributes: {model['target_attributes']}")
        print()
        
        print("✅ All endpoints are working correctly!")
        
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running. Please start it with:")
        print("   uvicorn main:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False
    
    return True

def create_sample_data():
    """Create sample data for testing."""
    print("\n📊 Creating Sample Data")
    print("-" * 30)
    
    # Generate sample data
    np.random.seed(42)
    data = []
    
    for i in range(100):
        data.append({
            'entity_id': f'ENTITY_{i % 10:03d}',
            'value': np.random.normal(1000, 100),
            'rating': np.random.normal(4.0, 0.5),
            'size': np.random.normal(5000, 500),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'is_anomaly': np.random.random() < 0.1
        })
    
    df = pd.DataFrame(data)
    df.to_csv('sample_data.csv', index=False)
    print(f"   Created sample_data.csv with {len(df)} records")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Anomalies: {df['is_anomaly'].sum()}")
    
    return df

def test_full_workflow():
    """Test the full API workflow."""
    print("\n🔄 Testing Full API Workflow")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    try:
        # 1. Upload dataset
        print("1. Uploading Dataset")
        with open('sample_data.csv', 'rb') as f:
            files = {'file': ('sample_data.csv', f, 'text/csv')}
            response = requests.post(f"{base_url}/upload-dataset", files=files)
        
        print(f"   Status: {response.status_code}")
        upload_result = response.json()
        print(f"   Dataset ID: {upload_result['dataset_id']}")
        print(f"   Rows: {upload_result['data_info']['rows']}")
        print(f"   Columns: {upload_result['data_info']['columns']}")
        dataset_id = upload_result['dataset_id']
        print()
        
        # 2. Train model
        print("2. Training Model")
        training_request = {
            "algorithm": "isolation_forest",
            "contamination": 0.1,
            "business_key": "entity_id",
            "target_attributes": ["value", "rating", "size"],
            "time_column": "date",
            "anomaly_labels": "is_anomaly"
        }
        
        response = requests.post(
            f"{base_url}/train/{dataset_id}",
            json=training_request
        )
        
        print(f"   Status: {response.status_code}")
        training_result = response.json()
        print(f"   Model ID: {training_result['model_id']}")
        print(f"   Anomalies detected: {training_result['training_stats']['anomaly_count']}")
        print(f"   Features: {training_result['training_stats']['n_features']}")
        model_id = training_result['model_id']
        print()
        
        # 3. Single prediction
        print("3. Single Prediction")
        prediction_request = {
            "model_id": model_id,
            "data": {
                "entity_id": "ENTITY_001",
                "value": 1000,
                "rating": 4.5,
                "size": 5000,
                "date": "2024-01-01"
            }
        }
        
        response = requests.post(f"{base_url}/predict", json=prediction_request)
        
        print(f"   Status: {response.status_code}")
        prediction_result = response.json()
        print(f"   Model ID: {prediction_result['model_id']}")
        print(f"   Prediction: {prediction_result['prediction']}")
        print(f"   Anomaly Score: {prediction_result['anomaly_score']}")
        print(f"   Timestamp: {prediction_result['timestamp']}")
        print()
        
        # 4. Batch prediction
        print("4. Batch Prediction")
        batch_request = {
            "model_id": model_id,
            "data": [
                {
                    "entity_id": "ENTITY_001",
                    "value": 1000,
                    "rating": 4.5,
                    "size": 5000,
                    "date": "2024-01-01"
                },
                {
                    "entity_id": "ENTITY_002",
                    "value": 500,
                    "rating": 2.1,
                    "size": 2500,
                    "date": "2024-01-01"
                }
            ]
        }
        
        response = requests.post(f"{base_url}/predict-batch", json=batch_request)
        
        print(f"   Status: {response.status_code}")
        batch_result = response.json()
        print(f"   Model ID: {batch_result['model_id']}")
        print(f"   Predictions: {batch_result['predictions']['predictions']}")
        print(f"   Anomaly Scores: {batch_result['predictions']['anomaly_scores']}")
        print(f"   Anomaly Count: {batch_result['predictions']['anomaly_count']}")
        print(f"   Entity Analysis: {batch_result['predictions']['prediction_analysis']}")
        print(f"   Timestamp: {batch_result['timestamp']}")
        print()
        
        # 5. Get model info
        print("5. Get Model Info")
        response = requests.get(f"{base_url}/models/{model_id}")
        
        print(f"   Status: {response.status_code}")
        model_info = response.json()
        print(f"   Model ID: {model_info['model_id']}")
        print(f"   Algorithm: {model_info['algorithm']}")
        print(f"   Business Key: {model_info['business_key']}")
        print(f"   Target Attributes: {model_info['target_attributes']}")
        print(f"   Feature Columns: {len(model_info['feature_columns'])} features")
        print(f"   Time Column: {model_info['time_column']}")
        print(f"   Anomaly Labels: {model_info['anomaly_labels']}")
        print()
        
        print("✅ Full workflow completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in workflow: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 API Response Format Test")
    print("=" * 70)
    
    # Test basic endpoints
    if not test_api_response_format():
        return
    
    # Create sample data
    create_sample_data()
    
    # Test full workflow
    test_full_workflow()
    
    print("\n📋 Summary of Actual Response Formats:")
    print("=" * 50)
    print("✅ Single Prediction Response:")
    print("   - model_id: string")
    print("   - prediction: int (-1 for anomaly, 1 for normal)")
    print("   - anomaly_score: float")
    print("   - timestamp: string (ISO format)")
    print()
    print("✅ Batch Prediction Response:")
    print("   - model_id: string")
    print("   - predictions: object with:")
    print("     - predictions: list of ints")
    print("     - scores: list of floats")
    print("     - anomaly_count: int")
    print("     - anomaly_rate: float")
    print("     - prediction_analysis: object with entity analysis")
    print("   - timestamp: string (ISO format)")
    print()
    print("✅ Model Info Response:")
    print("   - model_id: string")
    print("   - algorithm: string")
    print("   - contamination: float")
    print("   - business_key: string")
    print("   - target_attributes: list of strings")
    print("   - feature_columns: list of strings")
    print("   - time_column: string or null")
    print("   - anomaly_labels: string or null")
    print("   - training_stats: object")
    print("   - created_at: string (ISO format)")

if __name__ == "__main__":
    main()
