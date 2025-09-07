#!/usr/bin/env python3
"""
Test Date Serialization in APIs
Verifies that datetime objects are properly handled in API responses
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from generic_anomaly_detector import GenericAnomalyDetector

def test_date_serialization():
    """Test that dates are properly serialized in various scenarios"""
    print("🧪 Testing Date Serialization")
    print("=" * 50)
    
    # Create test data with various date formats
    test_data = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(10):
        test_data.append({
            'entity_id': f'ENTITY_{i:03d}',
            'value': np.random.uniform(100, 1000),
            'rating': np.random.uniform(1, 5),
            'date': base_date + timedelta(days=i),
            'timestamp': pd.Timestamp(base_date + timedelta(days=i, hours=i)),
            'is_anomaly': i % 3 == 0
        })
    
    df = pd.DataFrame(test_data)
    print(f"Created test data with {len(df)} records")
    print(f"Date column type: {df['date'].dtype}")
    print(f"Timestamp column type: {df['timestamp'].dtype}")
    
    # Test 1: Generic Anomaly Detector with dates
    print("\n1️⃣ Testing Generic Anomaly Detector with dates")
    print("-" * 40)
    
    try:
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        detector.train(
            df=df,
            business_key='entity_id',
            target_attributes=['value', 'rating'],
            time_column='date'
        )
        
        # Make predictions
        predictions = detector.predict(df, business_key='entity_id', time_column='date')
        print("✅ Generic detector works with dates")
        print(f"   Predictions shape: {len(predictions['predictions'])}")
        print(f"   Anomaly count: {predictions['anomaly_count']}")
        
    except Exception as e:
        print(f"❌ Generic detector failed: {str(e)}")
    
    # Test 2: JSON serialization of dates
    print("\n2️⃣ Testing JSON serialization of dates")
    print("-" * 40)
    
    try:
        # Test various date formats
        test_objects = {
            'datetime': datetime.now(),
            'pandas_timestamp': pd.Timestamp.now(),
            'numpy_datetime': np.datetime64('2024-01-01'),
            'date_string': '2024-01-01',
            'iso_string': datetime.now().isoformat()
        }
        
        # Custom serializer function
        def serialize_dates(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, np.datetime64):
                return pd.Timestamp(obj).strftime('%Y-%m-%d %H:%M:%S')
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        # Test serialization
        for key, value in test_objects.items():
            try:
                serialized = json.dumps({key: value}, default=serialize_dates)
                print(f"✅ {key}: {serialized}")
            except Exception as e:
                print(f"❌ {key}: {str(e)}")
        
    except Exception as e:
        print(f"❌ JSON serialization failed: {str(e)}")
    
    # Test 3: DataFrame with mixed date types
    print("\n3️⃣ Testing DataFrame with mixed date types")
    print("-" * 40)
    
    try:
        # Create DataFrame with various date formats
        mixed_df = pd.DataFrame({
            'entity_id': ['A', 'B', 'C'],
            'value': [100, 200, 300],
            'date_str': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'date_datetime': [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            'date_timestamp': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03')]
        })
        
        print("Original DataFrame:")
        print(mixed_df.dtypes)
        
        # Convert all date columns to consistent format
        for col in mixed_df.columns:
            if 'date' in col:
                mixed_df[col] = pd.to_datetime(mixed_df[col])
                print(f"Converted {col} to: {mixed_df[col].dtype}")
        
        print("\nAfter conversion:")
        print(mixed_df.dtypes)
        
        # Test with generic detector
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        detector.train(
            df=mixed_df,
            business_key='entity_id',
            target_attributes=['value'],
            time_column='date_str'
        )
        
        predictions = detector.predict(mixed_df, business_key='entity_id', time_column='date_str')
        print("✅ Mixed date types work with generic detector")
        
    except Exception as e:
        print(f"❌ Mixed date types failed: {str(e)}")
    
    # Test 4: API response simulation
    print("\n4️⃣ Testing API response simulation")
    print("-" * 40)
    
    try:
        # Simulate API response with dates
        api_response = {
            'model_id': 'test_model',
            'predictions': predictions['predictions'].tolist() if hasattr(predictions['predictions'], 'tolist') else list(predictions['predictions']),
            'anomaly_scores': predictions.get('anomaly_scores', []).tolist() if hasattr(predictions.get('anomaly_scores', []), 'tolist') else list(predictions.get('anomaly_scores', [])),
            'timestamp': datetime.now(),
            'data_info': {
                'datetime_columns': ['date', 'timestamp'],
                'sample_dates': df['date'].head(3).tolist()
            }
        }
        
        # Serialize for JSON response
        serialized_response = json.dumps(api_response, default=serialize_dates)
        print("✅ API response serialization successful")
        print(f"   Response length: {len(serialized_response)} characters")
        
        # Parse back to verify
        parsed_response = json.loads(serialized_response)
        print("✅ API response parsing successful")
        print(f"   Timestamp: {parsed_response['timestamp']}")
        
    except Exception as e:
        print(f"❌ API response simulation failed: {str(e)}")
    
    print("\n✅ Date serialization tests completed!")

def test_timezone_handling():
    """Test timezone handling in date processing"""
    print("\n🌍 Testing Timezone Handling")
    print("=" * 50)
    
    try:
        # Create data with timezone-aware dates
        df_tz = pd.DataFrame({
            'entity_id': ['A', 'B', 'C'],
            'value': [100, 200, 300],
            'date_utc': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']).tz_localize('UTC'),
            'date_est': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']).tz_localize('US/Eastern')
        })
        
        print("Original timezone-aware DataFrame:")
        print(df_tz.dtypes)
        
        # Test generic detector with timezone-aware dates
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        detector.train(
            df=df_tz,
            business_key='entity_id',
            target_attributes=['value'],
            time_column='date_utc'
        )
        
        predictions = detector.predict(df_tz, business_key='entity_id', time_column='date_utc')
        print("✅ Timezone-aware dates work with generic detector")
        
    except Exception as e:
        print(f"❌ Timezone handling failed: {str(e)}")

if __name__ == "__main__":
    test_date_serialization()
    test_timezone_handling()
    print("\n🎯 All date serialization tests completed!")
