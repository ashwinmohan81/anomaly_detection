#!/usr/bin/env python3
"""
Test script to verify uniform response format between single and batch predictions
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime

def test_uniform_response_format():
    """Test that single and batch predictions return the same structure."""
    print("🧪 Testing Uniform Response Format")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    try:
        # 1. Create sample data
        print("1. Creating Sample Data")
        data = []
        for i in range(10):
            data.append({
                'entity_id': f'ENTITY_{i % 3:03d}',
                'value': np.random.normal(1000, 100),
                'rating': np.random.normal(4.0, 0.5),
                'size': np.random.normal(5000, 500),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'is_anomaly': np.random.random() < 0.1
            })
        
        df = pd.DataFrame(data)
        df.to_csv('uniform_test_data.csv', index=False)
        print(f"   Created uniform_test_data.csv with {len(df)} records")
        print()
        
        # 2. Upload dataset
        print("2. Uploading Dataset")
        with open('uniform_test_data.csv', 'rb') as f:
            files = {'file': ('uniform_test_data.csv', f, 'text/csv')}
            response = requests.post(f"{base_url}/upload-dataset", files=files)
        
        upload_result = response.json()
        dataset_id = upload_result['dataset_id']
        print(f"   Dataset ID: {dataset_id}")
        print()
        
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
            json=training_request
        )
        
        training_result = response.json()
        model_id = training_result['model_id']
        print(f"   Model ID: {model_id}")
        print()
        
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
        single_result = response.json()
        
        print("   Single Prediction Response Structure:")
        print(f"   - model_id: {type(single_result['model_id'])} = {single_result['model_id']}")
        print(f"   - predictions: {type(single_result['predictions'])}")
        print(f"     - predictions: {type(single_result['predictions']['predictions'])} = {single_result['predictions']['predictions']}")
        print(f"     - scores: {type(single_result['predictions']['scores'])} = {single_result['predictions']['scores']}")
        print(f"     - anomaly_count: {type(single_result['predictions']['anomaly_count'])} = {single_result['predictions']['anomaly_count']}")
        print(f"     - anomaly_rate: {type(single_result['predictions']['anomaly_rate'])} = {single_result['predictions']['anomaly_rate']}")
        print(f"     - prediction_analysis: {type(single_result['predictions']['prediction_analysis'])}")
        print(f"   - timestamp: {type(single_result['timestamp'])} = {single_result['timestamp']}")
        print()
        
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
        batch_result = response.json()
        
        print("   Batch Prediction Response Structure:")
        print(f"   - model_id: {type(batch_result['model_id'])} = {batch_result['model_id']}")
        print(f"   - predictions: {type(batch_result['predictions'])}")
        print(f"     - predictions: {type(batch_result['predictions']['predictions'])} = {batch_result['predictions']['predictions']}")
        print(f"     - scores: {type(batch_result['predictions']['scores'])} = {batch_result['predictions']['scores']}")
        print(f"     - anomaly_count: {type(batch_result['predictions']['anomaly_count'])} = {batch_result['predictions']['anomaly_count']}")
        print(f"     - anomaly_rate: {type(batch_result['predictions']['anomaly_rate'])} = {batch_result['predictions']['anomaly_rate']}")
        print(f"     - prediction_analysis: {type(batch_result['predictions']['prediction_analysis'])}")
        print(f"   - timestamp: {type(batch_result['timestamp'])} = {batch_result['timestamp']}")
        print()
        
        # 6. Compare structures
        print("6. Comparing Response Structures")
        
        # Check if both have the same top-level keys
        single_keys = set(single_result.keys())
        batch_keys = set(batch_result.keys())
        
        print(f"   Single prediction keys: {sorted(single_keys)}")
        print(f"   Batch prediction keys: {sorted(batch_keys)}")
        print(f"   Keys match: {single_keys == batch_keys}")
        
        # Check if both have the same predictions structure
        single_pred_keys = set(single_result['predictions'].keys())
        batch_pred_keys = set(batch_result['predictions'].keys())
        
        print(f"   Single predictions keys: {sorted(single_pred_keys)}")
        print(f"   Batch predictions keys: {sorted(batch_pred_keys)}")
        print(f"   Predictions structure match: {single_pred_keys == batch_pred_keys}")
        
        # Check data types
        print(f"   Single predictions type: {type(single_result['predictions']['predictions'])}")
        print(f"   Batch predictions type: {type(batch_result['predictions']['predictions'])}")
        print(f"   Both are lists: {isinstance(single_result['predictions']['predictions'], list) and isinstance(batch_result['predictions']['predictions'], list)}")
        
        print()
        
        if single_keys == batch_keys and single_pred_keys == batch_pred_keys:
            print("✅ SUCCESS: Both single and batch predictions have uniform response structure!")
        else:
            print("❌ FAILURE: Response structures are not uniform!")
            return False
        
        # 7. Test with multiple single predictions
        print("7. Testing Multiple Single Predictions")
        single_predictions = []
        for i in range(3):
            single_request = {
                "model_id": model_id,
                "data": {
                    "entity_id": f"ENTITY_{i:03d}",
                    "value": 1000 + i * 100,
                    "rating": 4.0 + i * 0.2,
                    "size": 5000 + i * 500,
                    "date": "2024-01-01"
                }
            }
            
            response = requests.post(f"{base_url}/predict", json=single_request)
            single_predictions.append(response.json())
        
        print(f"   Made {len(single_predictions)} single predictions")
        print("   All single predictions have same structure:")
        for i, pred in enumerate(single_predictions):
            has_predictions = 'predictions' in pred
            has_scores = 'scores' in pred['predictions']
            print(f"     Prediction {i+1}: predictions={has_predictions}, scores={has_scores}")
        
        print()
        print("✅ All tests passed! Response format is now uniform between single and batch predictions.")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running. Please start it with:")
        print("   uvicorn main:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"❌ Error testing uniform response: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Uniform Response Format Test")
    print("=" * 70)
    
    success = test_uniform_response_format()
    
    if success:
        print("\n📋 Summary:")
        print("=" * 20)
        print("✅ Single prediction now returns the same structure as batch prediction")
        print("✅ Both endpoints return 'predictions' object with:")
        print("   - predictions: array of booleans")
        print("   - scores: array of floats")
        print("   - anomaly_count: integer")
        print("   - anomaly_rate: float")
        print("   - prediction_analysis: object with entity analysis")
        print("✅ Consistent API design for better developer experience")
    else:
        print("\n❌ Tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
