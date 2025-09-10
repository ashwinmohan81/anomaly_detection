#!/usr/bin/env python3
"""
Test script for mixed supervised/unsupervised anomaly detection
"""

import pandas as pd
import numpy as np
from enhanced_generic_anomaly_detector import EnhancedGenericAnomalyDetector
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def generate_mixed_data(n_samples=1000, n_entities=5):
    """Generate synthetic data with mixed supervised/unsupervised labels."""
    np.random.seed(42)
    
    # Generate base data
    data = []
    entities = [f'ENTITY_{i:03d}' for i in range(n_entities)]
    
    for i in range(n_samples):
        entity = np.random.choice(entities)
        
        # Generate normal data
        value = np.random.normal(1000, 100)
        rating = np.random.normal(4.0, 0.5)
        size = np.random.normal(5000, 500)
        
        # Add some anomalies
        is_anomaly = False
        if np.random.random() < 0.1:  # 10% anomalies
            is_anomaly = True
            value *= np.random.choice([0.3, 2.5])  # Very low or very high
            rating = np.random.choice([1.0, 5.0])  # Extreme ratings
            size *= np.random.choice([0.2, 3.0])   # Very small or very large
        
        # Create date
        date = datetime.now() - timedelta(days=np.random.randint(0, 365))
        
        # Determine if this sample should have a label
        has_label = np.random.random() < 0.3  # 30% of data has labels
        
        data.append({
            'entity_id': entity,
            'value': value,
            'rating': rating,
            'size': size,
            'date': date,
            'is_anomaly': is_anomaly if has_label else np.nan
        })
    
    df = pd.DataFrame(data)
    return df

def test_mixed_learning():
    """Test mixed supervised/unsupervised learning."""
    print("🧪 Testing Mixed Supervised/Unsupervised Learning")
    print("=" * 60)
    
    # Generate mixed data
    df = generate_mixed_data(1000, 5)
    
    print(f"📊 Generated {len(df)} samples")
    print(f"📈 Labeled samples: {df['is_anomaly'].notna().sum()}")
    print(f"📉 Unlabeled samples: {df['is_anomaly'].isna().sum()}")
    print(f"🎯 Labeled ratio: {df['is_anomaly'].notna().sum() / len(df):.2%}")
    
    # Test different algorithms
    algorithms = ['isolation_forest', 'random_forest', 'logistic_regression']
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\n🔧 Testing {algorithm.upper()}")
        print("-" * 40)
        
        # Train model
        detector = EnhancedGenericAnomalyDetector(algorithm=algorithm, contamination=0.1)
        
        try:
            detector.train(
                df=df,
                business_key='entity_id',
                target_attributes=['value', 'rating', 'size'],
                time_column='date',
                anomaly_labels='is_anomaly'
            )
            
            # Get model info
            info = detector.get_model_info()
            print(f"✅ Training successful")
            print(f"   Supervised data ratio: {info['supervised_data_ratio']:.2%}")
            print(f"   Has mixed data: {info['has_mixed_data']}")
            print(f"   Features: {info['training_stats']['n_features']}")
            print(f"   Anomalies detected: {info['training_stats']['anomaly_count']}")
            
            # Make predictions
            predictions = detector.predict(df, 'entity_id', 'date')
            print(f"   Predictions: {predictions['anomaly_count']} anomalies ({predictions['anomaly_rate']:.2%})")
            
            results[algorithm] = {
                'supervised_ratio': info['supervised_data_ratio'],
                'has_mixed_data': info['has_mixed_data'],
                'anomalies_detected': predictions['anomaly_count'],
                'anomaly_rate': predictions['anomaly_rate']
            }
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            results[algorithm] = {'error': str(e)}
    
    return results

def test_incremental_learning():
    """Test incremental learning with new labeled data."""
    print("\n🔄 Testing Incremental Learning")
    print("=" * 60)
    
    # Generate initial data (mostly unlabeled)
    df_initial = generate_mixed_data(500, 3)
    df_initial['is_anomaly'] = np.nan  # All unlabeled initially
    
    print(f"📊 Initial data: {len(df_initial)} samples (all unlabeled)")
    
    # Train initial model
    detector = EnhancedGenericAnomalyDetector(algorithm='random_forest', contamination=0.1)
    detector.train(
        df=df_initial,
        business_key='entity_id',
        target_attributes=['value', 'rating', 'size'],
        time_column='date',
        anomaly_labels='is_anomaly'
    )
    
    print(f"✅ Initial model trained (unsupervised)")
    
    # Generate new data with labels
    df_new = generate_mixed_data(200, 3)
    print(f"📊 New data: {len(df_new)} samples ({df_new['is_anomaly'].notna().sum()} labeled)")
    
    # Update model with new data
    try:
        detector.update_with_new_data(df_new, 'entity_id', 'is_anomaly')
        print(f"✅ Model updated with new data")
        
        # Get updated info
        info = detector.get_model_info()
        print(f"   Updated supervised ratio: {info['supervised_data_ratio']:.2%}")
        
    except Exception as e:
        print(f"❌ Update failed: {e}")

def test_scenarios():
    """Test different scenarios of mixed data."""
    print("\n🎯 Testing Different Scenarios")
    print("=" * 60)
    
    scenarios = [
        {"name": "All Unlabeled", "labeled_ratio": 0.0},
        {"name": "Few Labels (5%)", "labeled_ratio": 0.05},
        {"name": "Some Labels (20%)", "labeled_ratio": 0.20},
        {"name": "Many Labels (50%)", "labeled_ratio": 0.50},
        {"name": "All Labeled", "labeled_ratio": 1.0}
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print("-" * 30)
        
        # Generate data with specific labeled ratio
        df = generate_mixed_data(500, 3)
        
        # Set labels based on ratio
        n_labeled = int(len(df) * scenario['labeled_ratio'])
        if n_labeled > 0:
            labeled_indices = np.random.choice(len(df), n_labeled, replace=False)
            df.loc[~df.index.isin(labeled_indices), 'is_anomaly'] = np.nan
        
        print(f"   Labeled samples: {df['is_anomaly'].notna().sum()}/{len(df)}")
        
        # Test with Random Forest
        detector = EnhancedGenericAnomalyDetector(algorithm='random_forest', contamination=0.1)
        
        try:
            detector.train(
                df=df,
                business_key='entity_id',
                target_attributes=['value', 'rating', 'size'],
                time_column='date',
                anomaly_labels='is_anomaly'
            )
            
            info = detector.get_model_info()
            print(f"   ✅ Success: {info['supervised_data_ratio']:.2%} supervised")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

def main():
    """Main test function."""
    print("🚀 Enhanced Generic Anomaly Detector - Mixed Learning Test")
    print("=" * 70)
    
    # Test mixed learning
    results = test_mixed_learning()
    
    # Test incremental learning
    test_incremental_learning()
    
    # Test different scenarios
    test_scenarios()
    
    print("\n📊 Summary")
    print("=" * 30)
    for algorithm, result in results.items():
        if 'error' not in result:
            print(f"{algorithm}: {result['supervised_ratio']:.2%} supervised, {result['anomalies_detected']} anomalies")
        else:
            print(f"{algorithm}: ERROR - {result['error']}")
    
    print("\n✅ Mixed supervised/unsupervised learning test completed!")

if __name__ == "__main__":
    main()
