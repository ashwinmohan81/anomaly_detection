#!/usr/bin/env python3
"""
Simple API test using requests to test the actual running API
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import subprocess
import os
from datetime import datetime

def start_api_server():
    """Start the API server in the background"""
    try:
        # Try to start the server
        process = subprocess.Popen(
            ['python3', '-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8000'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)  # Wait for server to start
        return process
    except Exception as e:
        print(f"Failed to start server: {e}")
        return None

def test_api_endpoints():
    """Test all API endpoints"""
    print("🧪 Testing API Endpoints")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:8000"
    
    try:
        # Test health endpoint
        print("1. Testing Health Endpoint")
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        print()
        
        # Test root endpoint
        print("2. Testing Root Endpoint")
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        print()
        
        # Test algorithms endpoint
        print("3. Testing Algorithms Endpoint")
        response = requests.get(f"{base_url}/algorithms", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            algorithms = response.json()
            print(f"   Found {len(algorithms['algorithms'])} algorithms")
            for algo in algorithms['algorithms']:
                print(f"   - {algo['name']}: {algo['type']}")
        print()
        
        # Test examples endpoint
        print("4. Testing Examples Endpoint")
        response = requests.get(f"{base_url}/examples", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            examples = response.json()
            print(f"   Found {len(examples['examples'])} examples")
        print()
        
        # Test models endpoint
        print("5. Testing Models Endpoint")
        response = requests.get(f"{base_url}/models", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"   Found {models['count']} models")
        print()
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def test_full_workflow():
    """Test the complete workflow"""
    print("\n🔄 Testing Full Workflow")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:8000"
    
    try:
        # 1. Create and upload test data
        print("1. Creating Test Data")
        data = []
        for i in range(20):
            data.append({
                'entity_id': f'ENTITY_{i % 5:03d}',
                'value': np.random.normal(1000, 100),
                'rating': np.random.normal(4.0, 0.5),
                'size': np.random.normal(5000, 500),
                'date': (datetime.now()).strftime('%Y-%m-%d'),
                'is_anomaly': np.random.random() < 0.1
            })
        
        df = pd.DataFrame(data)
        df.to_csv('test_workflow_data.csv', index=False)
        print(f"   Created test_workflow_data.csv with {len(df)} records")
        
        # 2. Upload dataset
        print("2. Uploading Dataset")
        with open('test_workflow_data.csv', 'rb') as f:
            files = {'file': ('test_workflow_data.csv', f, 'text/csv')}
            response = requests.post(f"{base_url}/upload-dataset", files=files, timeout=10)
        
        if response.status_code != 200:
            print(f"   ❌ Upload failed: {response.status_code}")
            return False
            
        upload_result = response.json()
        dataset_id = upload_result['dataset_id']
        print(f"   ✅ Dataset uploaded: {dataset_id}")
        
        # 3. Train model
        print("3. Training Model")
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
            json=training_request,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"   ❌ Training failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
        training_result = response.json()
        model_id = training_result['model_id']
        print(f"   ✅ Model trained: {model_id}")
        print(f"   Anomalies detected: {training_result['training_stats']['anomaly_count']}")
        
        # 4. Single prediction
        print("4. Single Prediction")
        single_request = {
            "model_id": model_id,
            "data": {
                "entity_id": "ENTITY_001",
                "value": 1000,
                "rating": 4.5,
                "size": 5000,
                "date": "2024-01-01"
            }
        }
        
        response = requests.post(f"{base_url}/predict", json=single_request, timeout=10)
        
        if response.status_code != 200:
            print(f"   ❌ Single prediction failed: {response.status_code}")
            return False
            
        single_result = response.json()
        print(f"   ✅ Single prediction successful")
        print(f"   Prediction: {single_result['predictions']['predictions_by_entity']}")
        print(f"   Score: {single_result['predictions']['scores_by_entity']}")
        
        # 5. Batch prediction
        print("5. Batch Prediction")
        batch_request = {
            "model_id": model_id,
            "data": [
                {"entity_id": "ENTITY_001", "value": 1000, "rating": 4.5, "size": 5000, "date": "2024-01-01"},
                {"entity_id": "ENTITY_002", "value": 800, "rating": 3.2, "size": 3000, "date": "2024-01-01"},
                {"entity_id": "ENTITY_003", "value": 1200, "rating": 4.8, "size": 6000, "date": "2024-01-01"}
            ]
        }
        
        response = requests.post(f"{base_url}/predict-batch", json=batch_request, timeout=10)
        
        if response.status_code != 200:
            print(f"   ❌ Batch prediction failed: {response.status_code}")
            return False
            
        batch_result = response.json()
        print(f"   ✅ Batch prediction successful")
        print(f"   Predictions: {batch_result['predictions_by_entity']}")
        print(f"   Anomaly count: {batch_result['anomaly_count']}")
        print(f"   Anomaly rate: {batch_result['anomaly_rate']}")
        
        # 6. Get model info
        print("6. Get Model Info")
        response = requests.get(f"{base_url}/models/{model_id}", timeout=10)
        
        if response.status_code != 200:
            print(f"   ❌ Get model info failed: {response.status_code}")
            return False
            
        model_info = response.json()
        print(f"   ✅ Model info retrieved")
        print(f"   Algorithm: {model_info['algorithm']}")
        print(f"   Features: {len(model_info['feature_columns'])}")
        
        # Clean up
        if os.path.exists('test_workflow_data.csv'):
            os.remove('test_workflow_data.csv')
        
        print("\n✅ Full workflow completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in workflow: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Comprehensive API Testing")
    print("=" * 70)
    
    # Start API server
    print("Starting API server...")
    process = start_api_server()
    
    if process is None:
        print("❌ Failed to start API server")
        return False
    
    try:
        # Test basic endpoints
        basic_success = test_api_endpoints()
        
        # Test full workflow
        workflow_success = test_full_workflow()
        
        # Summary
        print("\n📊 Test Summary")
        print("=" * 50)
        print(f"Basic Endpoints: {'✅ PASS' if basic_success else '❌ FAIL'}")
        print(f"Full Workflow: {'✅ PASS' if workflow_success else '❌ FAIL'}")
        
        if basic_success and workflow_success:
            print("\n🎉 All tests passed! The API is working correctly.")
            return True
        else:
            print("\n❌ Some tests failed.")
            return False
            
    finally:
        # Clean up
        if process:
            process.terminate()
            process.wait()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
