#!/usr/bin/env python3
"""
Test script to demonstrate entity-grouped response format for large datasets
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
        process = subprocess.Popen(
            ['python3', '-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8000'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)
        return process
    except Exception as e:
        print(f"Failed to start server: {e}")
        return None

def create_large_test_data():
    """Create a large test dataset with multiple entities"""
    print("📊 Creating Large Test Dataset")
    print("-" * 40)
    
    data = []
    entities = ['FUND_001', 'FUND_002', 'FUND_003', 'FUND_004', 'FUND_005']
    dates = [(datetime.now() - pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(20)]
    
    for entity in entities:
        for date in dates:
            # Create some anomalies
            is_anomaly = np.random.random() < 0.1
            
            if is_anomaly:
                value = np.random.normal(2000, 200)
                rating = np.random.normal(5.5, 0.3)
                size = np.random.normal(8000, 800)
            else:
                value = np.random.normal(1000, 100)
                rating = np.random.normal(4.0, 0.5)
                size = np.random.normal(5000, 500)
            
            data.append({
                'entity_id': entity,
                'value': value,
                'rating': rating,
                'size': size,
                'date': date,
                'is_anomaly': is_anomaly
            })
    
    df = pd.DataFrame(data)
    df.to_csv('large_test_data.csv', index=False)
    print(f"   Created large_test_data.csv with {len(df)} records")
    print(f"   Entities: {df['entity_id'].nunique()}")
    print(f"   Records per entity: {len(df) // df['entity_id'].nunique()}")
    print(f"   Anomalies: {df['is_anomaly'].sum()}")
    
    return df

def test_entity_grouped_response():
    """Test the new entity-grouped response format"""
    print("\n🧪 Testing Entity-Grouped Response Format")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:8000"
    
    try:
        # 1. Upload dataset
        print("1. Uploading Large Dataset")
        with open('large_test_data.csv', 'rb') as f:
            files = {'file': ('large_test_data.csv', f, 'text/csv')}
            response = requests.post(f"{base_url}/upload-dataset", files=files, timeout=10)
        
        if response.status_code != 200:
            print(f"   ❌ Upload failed: {response.status_code}")
            return False
            
        upload_result = response.json()
        dataset_id = upload_result['dataset_id']
        print(f"   ✅ Dataset uploaded: {dataset_id}")
        
        # 2. Train model
        print("\n2. Training Model")
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
            return False
            
        training_result = response.json()
        model_id = training_result['model_id']
        print(f"   ✅ Model trained: {model_id}")
        
        # 3. Test single prediction (entity-grouped)
        print("\n3. Single Prediction (Entity-Grouped)")
        single_request = {
            "model_id": model_id,
            "data": {
                "entity_id": "FUND_001",
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
        print(f"   Response structure:")
        print(f"     - predictions_by_entity: {list(single_result['predictions']['predictions_by_entity'].keys())}")
        print(f"     - scores_by_entity: {list(single_result['predictions']['scores_by_entity'].keys())}")
        print(f"     - anomaly_count: {single_result['predictions']['anomaly_count']}")
        print(f"     - anomaly_rate: {single_result['predictions']['anomaly_rate']}")
        
        # 4. Test batch prediction with all data (entity-grouped)
        print("\n4. Batch Prediction - All Data (Entity-Grouped)")
        
        # Load the data for batch prediction
        df = pd.read_csv('large_test_data.csv')
        batch_data = df.to_dict('records')
        
        batch_request = {
            "model_id": model_id,
            "data": batch_data,
            "include_predictions": True,
            "include_scores": True
        }
        
        response = requests.post(f"{base_url}/predict-batch", json=batch_request, timeout=30)
        
        if response.status_code != 200:
            print(f"   ❌ Batch prediction failed: {response.status_code}")
            return False
            
        batch_result = response.json()
        print(f"   ✅ Batch prediction successful")
        print(f"   Total records: {batch_result['total_records']}")
        print(f"   Anomaly count: {batch_result['anomaly_count']}")
        print(f"   Anomaly rate: {batch_result['anomaly_rate']:.2%}")
        
        # Show entity-grouped structure
        predictions_by_entity = batch_result['predictions_by_entity']
        scores_by_entity = batch_result['scores_by_entity']
        
        print(f"\n   Entity-Grouped Results:")
        for entity_id in predictions_by_entity.keys():
            preds = predictions_by_entity[entity_id]
            scores = scores_by_entity[entity_id]
            anomaly_count = sum(preds)
            print(f"     {entity_id}: {len(preds)} records, {anomaly_count} anomalies")
            print(f"       Predictions: {preds[:3]}{'...' if len(preds) > 3 else ''}")
            print(f"       Scores: {[f'{s:.3f}' for s in scores[:3]]}{'...' if len(scores) > 3 else ''}")
        
        # 5. Test summary-only response
        print("\n5. Summary-Only Response")
        summary_request = {
            "model_id": model_id,
            "data": batch_data,
            "summary_only": True
        }
        
        response = requests.post(f"{base_url}/predict-batch", json=summary_request, timeout=30)
        
        if response.status_code != 200:
            print(f"   ❌ Summary request failed: {response.status_code}")
            return False
            
        summary_result = response.json()
        print(f"   ✅ Summary response successful")
        print(f"   Response keys: {list(summary_result.keys())}")
        print(f"   Total records: {summary_result['total_records']}")
        print(f"   Anomaly count: {summary_result['anomaly_count']}")
        print(f"   Anomaly rate: {summary_result['anomaly_rate']:.2%}")
        print(f"   Has predictions_by_entity: {'predictions_by_entity' in summary_result}")
        print(f"   Has scores_by_entity: {'scores_by_entity' in summary_result}")
        
        # 6. Test pagination
        print("\n6. Pagination Test")
        pagination_request = {
            "model_id": model_id,
            "data": batch_data,
            "include_predictions": True,
            "include_scores": True,
            "page": 1,
            "page_size": 20
        }
        
        response = requests.post(f"{base_url}/predict-batch", json=pagination_request, timeout=30)
        
        if response.status_code != 200:
            print(f"   ❌ Pagination request failed: {response.status_code}")
            return False
            
        pagination_result = response.json()
        print(f"   ✅ Pagination response successful")
        print(f"   Page: {pagination_result['pagination']['page']}")
        print(f"   Page size: {pagination_result['pagination']['page_size']}")
        print(f"   Total pages: {pagination_result['pagination']['total_pages']}")
        print(f"   Has next: {pagination_result['pagination']['has_next']}")
        print(f"   Has previous: {pagination_result['pagination']['has_previous']}")
        
        # Clean up
        if os.path.exists('large_test_data.csv'):
            os.remove('large_test_data.csv')
        
        print("\n✅ All entity-grouped response tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Entity-Grouped Response Format Testing")
    print("=" * 70)
    
    # Create large test data
    create_large_test_data()
    
    # Start API server
    print("\nStarting API server...")
    process = start_api_server()
    
    if process is None:
        print("❌ Failed to start API server")
        return False
    
    try:
        # Test entity-grouped responses
        success = test_entity_grouped_response()
        
        if success:
            print("\n🎉 Entity-grouped response format is working perfectly!")
            print("\n📋 Benefits of Entity-Grouped Format:")
            print("   ✅ Predictions and scores organized by entity ID")
            print("   ✅ Easy to find results for specific entities")
            print("   ✅ Reduces response size for large datasets")
            print("   ✅ Maintains all summary statistics")
            print("   ✅ Supports pagination for very large datasets")
            print("   ✅ Summary-only option for quick overviews")
        
        return success
        
    finally:
        # Clean up
        if process:
            process.terminate()
            process.wait()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
