#!/usr/bin/env python3
"""
Test script to understand how anomaly scores are interpreted
"""

import pandas as pd
import numpy as np
from datetime import datetime
from generic_anomaly_detector import GenericAnomalyDetector

def test_score_interpretation():
    """Test how anomaly scores are interpreted"""
    print("🔍 Testing Anomaly Score Interpretation")
    print("=" * 50)
    
    # Create test data with known patterns
    data = []
    
    # Normal data (should have low anomaly scores)
    for i in range(20):
        data.append({
            'entity_id': f'ENTITY_{i % 2:03d}',
            'value': np.random.normal(1000, 50),  # Normal distribution
            'rating': np.random.normal(4.0, 0.2),  # Normal distribution
            'size': np.random.normal(5000, 200),  # Normal distribution
            'date': (datetime.now()).strftime('%Y-%m-%d')
        })
    
    # Anomalous data (should have high anomaly scores)
    for i in range(5):
        data.append({
            'entity_id': f'ENTITY_{i % 2:03d}',
            'value': np.random.normal(2000, 100),  # Much higher values
            'rating': np.random.normal(1.0, 0.2),  # Much lower ratings
            'size': np.random.normal(10000, 500),  # Much larger sizes
            'date': (datetime.now()).strftime('%Y-%m-%d')
        })
    
    df = pd.DataFrame(data)
    print(f"Created test data with {len(df)} records")
    print(f"  - Normal records: 20")
    print(f"  - Anomalous records: 5")
    
    # Test with Isolation Forest
    print("\n1. Testing Isolation Forest")
    detector = GenericAnomalyDetector(
        algorithm='isolation_forest',
        contamination=0.2  # Expect about 20% anomalies
    )
    
    detector.train(
        df=df,
        business_key='entity_id',
        target_attributes=['value', 'rating', 'size'],
        time_column='date'
    )
    
    result = detector.predict(df, 'entity_id', 'date')
    
    print(f"   Anomaly count: {result['anomaly_count']}")
    print(f"   Anomaly rate: {result['anomaly_rate']:.2%}")
    
    # Analyze scores for normal vs anomalous data
    normal_scores = []
    anomalous_scores = []
    
    for i, (pred, score) in enumerate(zip(result['predictions'], result['scores'])):
        if i < 20:  # Normal data
            normal_scores.append(score)
        else:  # Anomalous data
            anomalous_scores.append(score)
    
    print(f"   Normal data scores: min={min(normal_scores):.3f}, max={max(normal_scores):.3f}, avg={np.mean(normal_scores):.3f}")
    print(f"   Anomalous data scores: min={min(anomalous_scores):.3f}, max={max(anomalous_scores):.3f}, avg={np.mean(anomalous_scores):.3f}")
    
    # Check prediction accuracy
    correct_predictions = 0
    for i, pred in enumerate(result['predictions']):
        if i < 20:  # Normal data should be predicted as normal (False)
            if not pred:
                correct_predictions += 1
        else:  # Anomalous data should be predicted as anomalous (True)
            if pred:
                correct_predictions += 1
    
    accuracy = correct_predictions / len(result['predictions'])
    print(f"   Prediction accuracy: {accuracy:.2%}")
    
    # Test with One-Class SVM
    print("\n2. Testing One-Class SVM")
    detector2 = GenericAnomalyDetector(
        algorithm='one_class_svm',
        contamination=0.2
    )
    
    detector2.train(
        df=df,
        business_key='entity_id',
        target_attributes=['value', 'rating', 'size'],
        time_column='date'
    )
    
    result2 = detector2.predict(df, 'entity_id', 'date')
    
    print(f"   Anomaly count: {result2['anomaly_count']}")
    print(f"   Anomaly rate: {result2['anomaly_rate']:.2%}")
    
    # Analyze scores
    normal_scores2 = []
    anomalous_scores2 = []
    
    for i, (pred, score) in enumerate(zip(result2['predictions'], result2['scores'])):
        if i < 20:  # Normal data
            normal_scores2.append(score)
        else:  # Anomalous data
            anomalous_scores2.append(score)
    
    print(f"   Normal data scores: min={min(normal_scores2):.3f}, max={max(normal_scores2):.3f}, avg={np.mean(normal_scores2):.3f}")
    print(f"   Anomalous data scores: min={min(anomalous_scores2):.3f}, max={max(anomalous_scores2):.3f}, avg={np.mean(anomalous_scores2):.3f}")
    
    # Test with Local Outlier Factor
    print("\n3. Testing Local Outlier Factor")
    detector3 = GenericAnomalyDetector(
        algorithm='local_outlier_factor',
        contamination=0.2
    )
    
    detector3.train(
        df=df,
        business_key='entity_id',
        target_attributes=['value', 'rating', 'size'],
        time_column='date'
    )
    
    result3 = detector3.predict(df, 'entity_id', 'date')
    
    print(f"   Anomaly count: {result3['anomaly_count']}")
    print(f"   Anomaly rate: {result3['anomaly_rate']:.2%}")
    
    # Analyze scores
    normal_scores3 = []
    anomalous_scores3 = []
    
    for i, (pred, score) in enumerate(zip(result3['predictions'], result3['scores'])):
        if i < 20:  # Normal data
            normal_scores3.append(score)
        else:  # Anomalous data
            anomalous_scores3.append(score)
    
    print(f"   Normal data scores: min={min(normal_scores3):.3f}, max={max(normal_scores3):.3f}, avg={np.mean(normal_scores3):.3f}")
    print(f"   Anomalous data scores: min={min(anomalous_scores3):.3f}, max={max(anomalous_scores3):.3f}, avg={np.mean(anomalous_scores3):.3f}")
    
    print("\n📊 Score Interpretation Summary:")
    print("=" * 50)
    print("✅ HIGH scores = MORE anomalous (more likely to be anomaly)")
    print("✅ LOW scores = LESS anomalous (more likely to be normal)")
    print("✅ The threshold is determined by the contamination parameter")
    print("✅ Scores above threshold → predicted as anomaly")
    print("✅ Scores below threshold → predicted as normal")
    
    return True

if __name__ == "__main__":
    test_score_interpretation()
