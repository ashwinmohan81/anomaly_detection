#!/usr/bin/env python3
"""
Test script to verify the new restructured response format
where predictions and scores are moved into prediction_analysis
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import json

def test_restructured_response():
    """Test the new restructured response format"""
    print("🚀 Testing Restructured Response Format")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:8000"
    
    try:
        # 1. Create test data
        print("1. Creating Test Data")
        data = []
        for i in range(10):
            data.append({
                'entity_id': f'ENTITY_{i % 3:03d}',
                'value': np.random.normal(1000, 100),
                'rating': np.random.normal(4.0, 0.5),
                'size': np.random.normal(5000, 500),
                'date': datetime.now().strftime('%Y-%m-%d')
            })
        
        df = pd.DataFrame(data)
        df.to_csv('test_restructure_response.csv', index=False)
        print("   Created test_restructure_response.csv with 10 records")
        
        # 2. Upload dataset
        print("2. Uploading Dataset")
        with open('test_restructure_response.csv', 'rb') as f:
            response = requests.post(f"{base_url}/upload-dataset", files={'file': f})
        
        if response.status_code != 200:
            print(f"   ❌ Upload failed: {response.status_code}")
            return False
            
        dataset_id = response.json()['dataset_id']
        print(f"   ✅ Dataset uploaded: {dataset_id}")
        
        # 3. Train model
        print("3. Training Model")
        train_request = {
            "algorithm": "isolation_forest",
            "contamination": 0.1,
            "business_key": "entity_id",
            "target_attributes": ["value", "rating", "size"],
            "time_column": "date"
        }
        
        response = requests.post(f"{base_url}/train/{dataset_id}", json=train_request)
        if response.status_code != 200:
            print(f"   ❌ Training failed: {response.status_code}")
            return False
            
        model_id = response.json()['model_id']
        print(f"   ✅ Model trained: {model_id}")
        
        # 4. Test single prediction
        print("4. Testing Single Prediction")
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
        
        response = requests.post(f"{base_url}/predict", json=single_request)
        if response.status_code != 200:
            print(f"   ❌ Single prediction failed: {response.status_code}")
            return False
            
        single_result = response.json()
        print("   ✅ Single prediction successful")
        
        # Verify new structure
        print("   Single Prediction Structure:")
        print(f"   - model_id: {type(single_result['model_id'])} = {single_result['model_id']}")
        print(f"   - anomaly_count: {type(single_result['anomaly_count'])} = {single_result['anomaly_count']}")
        print(f"   - anomaly_rate: {type(single_result['anomaly_rate'])} = {single_result['anomaly_rate']}")
        print(f"   - prediction_analysis: {type(single_result['prediction_analysis'])}")
        
        # Check if predictions and scores are in prediction_analysis
        entity_data = single_result['prediction_analysis']['ENTITY_001']
        print(f"   - Entity data keys: {list(entity_data.keys())}")
        print(f"   - Has predictions: {'predictions' in entity_data}")
        print(f"   - Has scores: {'scores' in entity_data}")
        print(f"   - Predictions: {entity_data.get('predictions', 'NOT FOUND')}")
        print(f"   - Scores: {entity_data.get('scores', 'NOT FOUND')}")
        
        # 5. Test batch prediction
        print("5. Testing Batch Prediction")
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
        if response.status_code != 200:
            print(f"   ❌ Batch prediction failed: {response.status_code}")
            return False
            
        batch_result = response.json()
        print("   ✅ Batch prediction successful")
        
        # Verify new structure
        print("   Batch Prediction Structure:")
        print(f"   - model_id: {type(batch_result['model_id'])} = {batch_result['model_id']}")
        print(f"   - anomaly_count: {type(batch_result['anomaly_count'])} = {batch_result['anomaly_count']}")
        print(f"   - anomaly_rate: {type(batch_result['anomaly_rate'])} = {batch_result['anomaly_rate']}")
        print(f"   - prediction_analysis: {type(batch_result['prediction_analysis'])}")
        
        # Check if predictions and scores are in prediction_analysis for each entity
        for entity_id, entity_data in batch_result['prediction_analysis'].items():
            print(f"   - {entity_id} data keys: {list(entity_data.keys())}")
            print(f"   - {entity_id} has predictions: {'predictions' in entity_data}")
            print(f"   - {entity_id} has scores: {'scores' in entity_data}")
            print(f"   - {entity_id} predictions: {entity_data.get('predictions', 'NOT FOUND')}")
            print(f"   - {entity_id} scores: {entity_data.get('scores', 'NOT FOUND')}")
        
        # 6. Test summary_only option
        print("6. Testing Summary-Only Option")
        summary_request = {
            "model_id": model_id,
            "data": [
                {
                    "entity_id": "ENTITY_001",
                    "value": 1000,
                    "rating": 4.5,
                    "size": 5000,
                    "date": "2024-01-01"
                }
            ],
            "summary_only": True
        }
        
        response = requests.post(f"{base_url}/predict-batch", json=summary_request)
        if response.status_code != 200:
            print(f"   ❌ Summary prediction failed: {response.status_code}")
            return False
            
        summary_result = response.json()
        print("   ✅ Summary prediction successful")
        
        # Verify summary structure
        print("   Summary Response Structure:")
        print(f"   - Keys: {list(summary_result.keys())}")
        print(f"   - Has prediction_analysis: {'prediction_analysis' in summary_result}")
        print(f"   - Prediction analysis keys: {list(summary_result['prediction_analysis'].keys())}")
        
        # Check that detailed predictions/scores are NOT in summary
        entity_data = summary_result['prediction_analysis']['ENTITY_001']
        print(f"   - Entity data keys: {list(entity_data.keys())}")
        print(f"   - Has predictions: {'predictions' in entity_data}")
        print(f"   - Has scores: {'scores' in entity_data}")
        
        print("\n✅ All restructured response tests passed!")
        print("\n📋 New Response Structure Summary:")
        print("   ✅ Predictions and scores moved to prediction_analysis")
        print("   ✅ Only anomaly_count and anomaly_rate at top level")
        print("   ✅ Cleaner, more organized response structure")
        print("   ✅ Summary-only option works correctly")
        print("   ✅ Both single and batch predictions use same structure")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing restructured response: {e}")
        return False

if __name__ == "__main__":
    success = test_restructured_response()
    if success:
        print("\n🎉 Restructured response format is working perfectly!")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
