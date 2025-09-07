#!/usr/bin/env python3
"""
Test Generic Anomaly Detection Engine
Works with any business keys and attributes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from generic_anomaly_detector import GenericAnomalyDetector
import json

def create_stock_data():
    """Create sample stock data."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends
    
    instruments = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    all_data = []
    
    for instrument in instruments:
        base_price = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500, 'AMZN': 3000, 'TSLA': 200}[instrument]
        
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        volume = np.random.lognormal(15, 0.5, len(dates))
        
        for date, price, vol in zip(dates, prices, volume):
            all_data.append({
                'date': date,
                'symbol': instrument,
                'close': price,
                'volume': int(vol)
            })
    
    return pd.DataFrame(all_data)

def create_index_data():
    """Create sample index data."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    dates = dates[dates.weekday < 5]
    
    indices = ['SPX', 'NDX', 'RUT', 'VIX', 'DXY']
    all_data = []
    
    for index in indices:
        base_value = {'SPX': 4000, 'NDX': 12000, 'RUT': 2000, 'VIX': 20, 'DXY': 100}[index]
        
        returns = np.random.normal(0.0003, 0.015, len(dates))
        values = [base_value]
        for ret in returns[1:]:
            values.append(values[-1] * (1 + ret))
        
        for date, value in zip(dates, values):
            all_data.append({
                'date': date,
                'index_name': index,
                'value': value,
                'volume': np.random.lognormal(14, 0.3)
            })
    
    return pd.DataFrame(all_data)

def create_customer_data():
    """Create sample customer transaction data."""
    np.random.seed(42)
    
    customers = ['CUST_001', 'CUST_002', 'CUST_003', 'CUST_004', 'CUST_005']
    all_data = []
    
    for customer in customers:
        # Generate daily transaction data
        for i in range(365):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            
            # Normal transaction patterns
            base_amount = np.random.lognormal(6, 1)  # $100-1000 range
            base_frequency = np.random.poisson(3)  # 0-10 transactions per day
            
            all_data.append({
                'date': date,
                'customer_id': customer,
                'transaction_amount': base_amount,
                'transaction_count': base_frequency,
                'avg_transaction': base_amount / max(base_frequency, 1)
            })
    
    return pd.DataFrame(all_data)

def create_sensor_data():
    """Create sample IoT sensor data."""
    np.random.seed(42)
    
    sensors = ['SENSOR_001', 'SENSOR_002', 'SENSOR_003', 'SENSOR_004', 'SENSOR_005']
    all_data = []
    
    for sensor in sensors:
        # Generate daily sensor readings (much smaller dataset)
        for i in range(365):  # Daily for a year instead of hourly
            timestamp = datetime(2024, 1, 1) + timedelta(days=i)
            
            # Normal sensor readings
            temperature = 20 + 10 * np.sin(i * 2 * np.pi / 365) + np.random.normal(0, 2)
            humidity = 50 + 20 * np.sin(i * 2 * np.pi / 365) + np.random.normal(0, 5)
            pressure = 1013 + np.random.normal(0, 10)
            
            all_data.append({
                'timestamp': timestamp,
                'sensor_id': sensor,
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure
            })
    
    return pd.DataFrame(all_data)

def test_stock_anomaly_detection():
    """Test anomaly detection on stock data."""
    print("📈 Testing Stock Anomaly Detection")
    print("=" * 50)
    
    # Create stock data
    df = create_stock_data()
    print(f"Created {len(df)} records for {df['symbol'].nunique()} stocks")
    
    # Train model
    detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
    detector.train(
        df=df,
        business_key='symbol',
        target_attributes=['close', 'volume'],
        time_column='date'
    )
    
    print(f"✅ Model trained successfully!")
    print(f"   Business keys: {detector.business_keys}")
    print(f"   Target attributes: {detector.target_attributes}")
    print(f"   Features: {len(detector.feature_columns)}")
    print(f"   Anomalies: {detector.training_stats['anomaly_count']}")
    
    # Test predictions
    predictions = detector.predict(df, business_key='symbol', time_column='date')
    print(f"   Predictions: {predictions['anomaly_count']} anomalies ({predictions['anomaly_rate']:.1%})")
    
    # Test cross-entity anomalies
    cross_entity = detector.detect_cross_entity_anomalies(df, business_key='symbol', time_column='date')
    print(f"   Cross-entity events: {cross_entity['total_cross_entity_events']}")
    
    return detector

def test_index_anomaly_detection():
    """Test anomaly detection on index data."""
    print("\n📊 Testing Index Anomaly Detection")
    print("=" * 50)
    
    # Create index data
    df = create_index_data()
    print(f"Created {len(df)} records for {df['index_name'].nunique()} indices")
    
    # Train model
    detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
    detector.train(
        df=df,
        business_key='index_name',
        target_attributes=['value', 'volume'],
        time_column='date'
    )
    
    print(f"✅ Model trained successfully!")
    print(f"   Business keys: {detector.business_keys}")
    print(f"   Target attributes: {detector.target_attributes}")
    print(f"   Features: {len(detector.feature_columns)}")
    print(f"   Anomalies: {detector.training_stats['anomaly_count']}")
    
    # Test predictions
    predictions = detector.predict(df, business_key='index_name', time_column='date')
    print(f"   Predictions: {predictions['anomaly_count']} anomalies ({predictions['anomaly_rate']:.1%})")
    
    return detector

def test_customer_anomaly_detection():
    """Test anomaly detection on customer data."""
    print("\n👥 Testing Customer Anomaly Detection")
    print("=" * 50)
    
    # Create customer data
    df = create_customer_data()
    print(f"Created {len(df)} records for {df['customer_id'].nunique()} customers")
    
    # Train model
    detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
    detector.train(
        df=df,
        business_key='customer_id',
        target_attributes=['transaction_amount', 'transaction_count', 'avg_transaction'],
        time_column='date'
    )
    
    print(f"✅ Model trained successfully!")
    print(f"   Business keys: {detector.business_keys}")
    print(f"   Target attributes: {detector.target_attributes}")
    print(f"   Features: {len(detector.feature_columns)}")
    print(f"   Anomalies: {detector.training_stats['anomaly_count']}")
    
    # Test predictions
    predictions = detector.predict(df, business_key='customer_id', time_column='date')
    print(f"   Predictions: {predictions['anomaly_count']} anomalies ({predictions['anomaly_rate']:.1%})")
    
    return detector

def test_sensor_anomaly_detection():
    """Test anomaly detection on sensor data."""
    print("\n🌡️ Testing Sensor Anomaly Detection")
    print("=" * 50)
    
    # Create sensor data
    print("Creating sensor data...")
    df = create_sensor_data()
    print(f"Created {len(df)} records for {df['sensor_id'].nunique()} sensors")
    
    # Train model
    print("Training model...")
    detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
    detector.train(
        df=df,
        business_key='sensor_id',
        target_attributes=['temperature', 'humidity', 'pressure'],
        time_column='timestamp'
    )
    
    print(f"✅ Model trained successfully!")
    print(f"   Business keys: {detector.business_keys}")
    print(f"   Target attributes: {detector.target_attributes}")
    print(f"   Features: {len(detector.feature_columns)}")
    print(f"   Anomalies: {detector.training_stats['anomaly_count']}")
    
    # Test predictions
    print("Making predictions...")
    predictions = detector.predict(df, business_key='sensor_id', time_column='timestamp')
    print(f"   Predictions: {predictions['anomaly_count']} anomalies ({predictions['anomaly_rate']:.1%})")
    
    return detector

def test_feature_engineering():
    """Test the feature engineering capabilities."""
    print("\n🔧 Testing Feature Engineering")
    print("=" * 50)
    
    # Create simple test data
    df = create_stock_data()
    detector = GenericAnomalyDetector()
    
    # Create features
    features_df = detector._create_generic_features(
        df, 
        business_key='symbol', 
        target_attributes=['close', 'volume'],
        time_column='date'
    )
    
    print(f"✅ Features created successfully!")
    print(f"   Original columns: {len(df.columns)}")
    print(f"   Feature columns: {len(features_df.columns)}")
    
    # Show feature types
    entity_features = [col for col in features_df.columns if any(x in col for x in ['_change', '_ma_', '_std_', '_zscore_', '_volatility_', '_momentum_'])]
    cross_entity_features = [col for col in features_df.columns if col.startswith('cross_')]
    temporal_features = [col for col in features_df.columns if col in ['hour', 'day_of_week', 'month', 'quarter', 'is_weekend']]
    
    print(f"\n📊 Feature Categories:")
    print(f"   Entity-specific features: {len(entity_features)}")
    print(f"   Cross-entity features: {len(cross_entity_features)}")
    print(f"   Temporal features: {len(temporal_features)}")
    
    # Show some examples
    print(f"\n📈 Entity-Specific Features (first 10):")
    for feature in entity_features[:10]:
        print(f"   - {feature}")
    
    print(f"\n🔗 Cross-Entity Features (first 10):")
    for feature in cross_entity_features[:10]:
        print(f"   - {feature}")
    
    print(f"\n⏰ Temporal Features:")
    for feature in temporal_features:
        print(f"   - {feature}")

def compare_different_data_types():
    """Compare anomaly detection across different data types."""
    print("\n🔄 Comparing Different Data Types")
    print("=" * 60)
    
    data_types = [
        ("Stocks", create_stock_data(), 'symbol', ['close', 'volume'], 'date'),
        ("Indices", create_index_data(), 'index_name', ['value', 'volume'], 'date'),
        ("Customers", create_customer_data(), 'customer_id', ['transaction_amount', 'transaction_count'], 'date'),
        ("Sensors", create_sensor_data(), 'sensor_id', ['temperature', 'humidity', 'pressure'], 'timestamp')
    ]
    
    results = []
    
    for data_type, df, business_key, target_attrs, time_col in data_types:
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        detector.train(df, business_key, target_attrs, time_column=time_col)
        
        predictions = detector.predict(df, business_key, time_col)
        
        results.append({
            'data_type': data_type,
            'entities': len(detector.business_keys),
            'features': len(detector.feature_columns),
            'anomalies': predictions['anomaly_count'],
            'anomaly_rate': predictions['anomaly_rate']
        })
    
    print(f"\n📊 Comparison Results:")
    print(f"{'Data Type':<12} {'Entities':<8} {'Features':<8} {'Anomalies':<9} {'Rate':<8}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['data_type']:<12} {result['entities']:<8} {result['features']:<8} {result['anomalies']:<9} {result['anomaly_rate']:.1%}")

def main():
    """Main test workflow."""
    import sys
    
    # Check if user wants to skip sensor test
    skip_sensor = '--skip-sensor' in sys.argv
    
    try:
        print("🚀 Testing Generic Anomaly Detection Engine")
        print("=" * 60)
        print("This engine works with ANY business keys and attributes!")
        if skip_sensor:
            print("(Skipping sensor test for faster execution)")
        print()
        
        # Test different data types
        test_stock_anomaly_detection()
        test_index_anomaly_detection()
        test_customer_anomaly_detection()
        
        if not skip_sensor:
            test_sensor_anomaly_detection()
        else:
            print("\n🌡️ Skipping Sensor Test (use --skip-sensor to skip)")
        
        # Test feature engineering
        test_feature_engineering()
        
        # Compare different data types
        compare_different_data_types()
        
        print("\n🎉 Generic Anomaly Detection Test Completed!")
        print("💡 Key Benefits:")
        print("   ✅ Works with ANY business keys (symbols, indices, customers, sensors)")
        print("   ✅ Works with ANY attributes (prices, volumes, temperatures, transactions)")
        print("   ✅ Automatic feature engineering for all data types")
        print("   ✅ Cross-entity anomaly detection")
        print("   ✅ Temporal feature support")
        print("   ✅ Single generic solution for all use cases")
        print("\n💡 Usage: python test_generic_detector.py [--skip-sensor]")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
