#!/usr/bin/env python3
"""
Test script to validate anomaly detection API deployment
"""

import requests
import json
import time
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_health():
    """Test health endpoint."""
    print("🔍 Testing health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def test_upload():
    """Test dataset upload."""
    print("📤 Testing dataset upload...")
    
    # Create sample data
    sample_data = """date,price,volume
2024-01-01,100.0,1000000
2024-01-02,101.0,1100000
2024-01-03,99.0,900000
2024-01-04,102.0,1200000
2024-01-05,98.0,800000"""
    
    try:
        files = {'file': ('test_data.csv', sample_data, 'text/csv')}
        response = requests.post(f"{BASE_URL}/upload-dataset", files=files, timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Upload successful: Dataset ID {data['dataset_id']}")
            return data['dataset_id']
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Upload error: {e}")
        return None

def test_training(dataset_id):
    """Test model training."""
    print("🤖 Testing model training...")
    
    training_request = {
        "algorithm": "isolation_forest",
        "contamination": 0.1,
        "target_column": "price",
        "feature_columns": ["volume"]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/train/{dataset_id}",
            json=training_request,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Training successful: Model ID {data['model_id']}")
            return data['model_id']
        else:
            print(f"❌ Training failed: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Training error: {e}")
        return None

def test_prediction(model_id):
    """Test single prediction."""
    print("🔮 Testing single prediction...")
    
    prediction_request = {
        "model_id": model_id,
        "data": {
            "price": 150.0,
            "volume": 1000000
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=prediction_request,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            prediction = data['prediction']
            print(f"✅ Prediction successful:")
            print(f"   Is anomaly: {prediction['is_anomaly']}")
            print(f"   Score: {prediction['score']:.4f}")
            print(f"   Confidence: {prediction['confidence']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_batch_prediction(model_id):
    """Test batch prediction."""
    print("📊 Testing batch prediction...")
    
    batch_request = {
        "model_id": model_id,
        "data": [
            {"price": 100.0, "volume": 1000000},
            {"price": 200.0, "volume": 1000000},  # Potential anomaly
            {"price": 101.0, "volume": 1000000}
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict-batch",
            json=batch_request,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data['predictions']
            print(f"✅ Batch prediction successful:")
            print(f"   Total predictions: {len(predictions['predictions'])}")
            print(f"   Anomalies found: {predictions['anomaly_count']}")
            print(f"   Anomaly rate: {predictions['anomaly_rate']:.2%}")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Batch prediction error: {e}")
        return False

def test_models_list():
    """Test models listing."""
    print("📋 Testing models list...")
    
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models list successful: {data['count']} models found")
            for model in data['models']:
                print(f"   - {model['model_id']} ({model['algorithm']})")
            return True
        else:
            print(f"❌ Models list failed: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Models list error: {e}")
        return False

def test_metrics():
    """Test metrics endpoint."""
    print("📈 Testing metrics endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=TIMEOUT)
        
        if response.status_code == 200:
            print("✅ Metrics endpoint accessible")
            return True
        else:
            print(f"❌ Metrics failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Metrics error: {e}")
        return False

def run_performance_test():
    """Run basic performance test."""
    print("⚡ Running performance test...")
    
    start_time = time.time()
    success_count = 0
    total_requests = 10
    
    for i in range(total_requests):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                success_count += 1
        except:
            pass
    
    end_time = time.time()
    duration = end_time - start_time
    success_rate = (success_count / total_requests) * 100
    avg_response_time = duration / total_requests
    
    print(f"✅ Performance test completed:")
    print(f"   Success rate: {success_rate:.1f}%")
    print(f"   Average response time: {avg_response_time:.3f}s")
    print(f"   Total duration: {duration:.3f}s")
    
    return success_rate >= 90 and avg_response_time < 1.0

def main():
    """Run all tests."""
    print("🚀 Starting Anomaly Detection API Tests")
    print("=" * 50)
    print(f"Testing URL: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    tests = [
        ("Health Check", test_health),
        ("Metrics Endpoint", test_metrics),
        ("Performance Test", run_performance_test),
    ]
    
    # Run basic tests first
    basic_tests_passed = 0
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        if test_func():
            basic_tests_passed += 1
        else:
            print(f"❌ {test_name} failed - stopping tests")
            sys.exit(1)
    
    print(f"\n✅ Basic tests passed: {basic_tests_passed}/{len(tests)}")
    
    # Run API workflow tests
    print("\n🔄 Testing API Workflow...")
    
    # Upload dataset
    dataset_id = test_upload()
    if not dataset_id:
        print("❌ Cannot proceed without dataset upload")
        sys.exit(1)
    
    # Train model
    model_id = test_training(dataset_id)
    if not model_id:
        print("❌ Cannot proceed without model training")
        sys.exit(1)
    
    # Test predictions
    prediction_tests = [
        ("Single Prediction", lambda: test_prediction(model_id)),
        ("Batch Prediction", lambda: test_batch_prediction(model_id)),
        ("Models List", test_models_list),
    ]
    
    api_tests_passed = 0
    for test_name, test_func in prediction_tests:
        print(f"\n🧪 {test_name}")
        if test_func():
            api_tests_passed += 1
        else:
            print(f"⚠️  {test_name} failed")
    
    print(f"\n✅ API tests passed: {api_tests_passed}/{len(prediction_tests)}")
    
    # Final summary
    total_tests = len(tests) + len(prediction_tests)
    total_passed = basic_tests_passed + api_tests_passed
    
    print("\n" + "=" * 50)
    print("🎉 Test Summary")
    print("=" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success rate: {(total_passed/total_tests)*100:.1f}%")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Deployment is working correctly.")
        sys.exit(0)
    else:
        print(f"\n❌ {total_tests - total_passed} tests failed. Check deployment.")
        sys.exit(1)

if __name__ == "__main__":
    main()
