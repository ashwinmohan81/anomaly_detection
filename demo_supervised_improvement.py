#!/usr/bin/env python3
"""
Demonstration: How Supervised Learning Improves Anomaly Detection
Shows practical example with fund rating data
"""

import pandas as pd
import numpy as np
from generic_anomaly_detector import GenericAnomalyDetector
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def create_fund_example():
    """Create a realistic fund rating example with known anomalies"""
    print("📊 Creating Fund Rating Example with Known Anomalies")
    print("=" * 60)
    
    # Create sample fund data
    np.random.seed(42)
    funds = ['FUND_A', 'FUND_B', 'FUND_C', 'FUND_D', 'FUND_E']
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    
    all_data = []
    
    for fund in funds:
        base_rating = np.random.uniform(3, 4.5)
        base_value = np.random.uniform(1000000, 5000000)
        
        for i, date in enumerate(dates):
            # Normal fund behavior
            rating = base_rating + np.random.normal(0, 0.2)
            value = base_value * (1 + np.random.normal(0, 0.05))
            size = base_value / 1000 + np.random.normal(0, 100)
            returns = np.random.normal(0.001, 0.02)
            
            # Inject specific anomalies
            is_anomaly = False
            
            # Fund A: Rating drops dramatically on specific dates
            if fund == 'FUND_A' and date.strftime('%Y-%m-%d') in ['2024-02-15', '2024-03-10']:
                rating = 1.5  # Extreme low rating
                is_anomaly = True
            
            # Fund B: Value crashes on specific dates
            if fund == 'FUND_B' and date.strftime('%Y-%m-%d') in ['2024-02-20', '2024-03-05']:
                value *= 0.3  # 70% value drop
                rating = max(1, rating - 1.5)  # Rating also drops
                is_anomaly = True
            
            # Fund C: Sudden rating spike (suspicious)
            if fund == 'FUND_C' and date.strftime('%Y-%m-%d') in ['2024-02-25', '2024-03-15']:
                rating = 5.0  # Perfect rating
                value *= 1.5  # Value also spikes
                is_anomaly = True
            
            # Fund D: Extreme returns
            if fund == 'FUND_D' and date.strftime('%Y-%m-%d') in ['2024-02-28', '2024-03-12']:
                returns = np.random.choice([-0.4, 0.6])  # ±40-60% returns
                is_anomaly = True
            
            # Fund E: Size explosion (suspicious growth)
            if fund == 'FUND_E' and date.strftime('%Y-%m-%d') in ['2024-03-01', '2024-03-20']:
                size *= 10  # 10x size increase
                is_anomaly = True
            
            all_data.append({
                'date': date,
                'fund_id': fund,
                'fund_rating': max(1, min(5, round(rating, 1))),
                'fund_value': max(0, int(value)),
                'fund_size': max(0, int(size)),
                'fund_returns': returns,
                'is_anomaly': is_anomaly
            })
    
    df = pd.DataFrame(all_data)
    return df

def test_unsupervised_vs_supervised(df):
    """Compare unsupervised vs supervised performance"""
    print("\n🔍 TESTING UNSUPERVISED vs SUPERVISED")
    print("=" * 50)
    
    # Test unsupervised
    print("\n1️⃣ UNSUPERVISED LEARNING (No Labels)")
    print("-" * 40)
    
    detector_unsupervised = GenericAnomalyDetector(
        algorithm='isolation_forest', 
        contamination=0.1
    )
    detector_unsupervised.train(
        df=df,
        business_key='fund_id',
        target_attributes=['fund_rating', 'fund_value', 'fund_size', 'fund_returns'],
        time_column='date'
    )
    
    predictions_unsupervised = detector_unsupervised.predict(df, business_key='fund_id', time_column='date')
    
    # Test supervised
    print("\n2️⃣ SUPERVISED LEARNING (With True Labels)")
    print("-" * 40)
    
    detector_supervised = GenericAnomalyDetector(
        algorithm='random_forest', 
        contamination=0.1
    )
    detector_supervised.train(
        df=df,
        business_key='fund_id',
        target_attributes=['fund_rating', 'fund_value', 'fund_size', 'fund_returns'],
        time_column='date',
        anomaly_labels='is_anomaly'  # Provide true labels!
    )
    
    predictions_supervised = detector_supervised.predict(df, business_key='fund_id', time_column='date')
    
    # Compare results
    y_true = df['is_anomaly'].astype(int)
    y_pred_unsupervised = predictions_unsupervised['predictions']
    y_pred_supervised = predictions_supervised['predictions']
    
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    # Unsupervised metrics
    cm_unsupervised = confusion_matrix(y_true, y_pred_unsupervised)
    tn_u, fp_u, fn_u, tp_u = cm_unsupervised.ravel()
    accuracy_u = (tp_u + tn_u) / (tp_u + tn_u + fp_u + fn_u)
    precision_u = tp_u / (tp_u + fp_u) if (tp_u + fp_u) > 0 else 0
    recall_u = tp_u / (tp_u + fn_u) if (tp_u + fn_u) > 0 else 0
    f1_u = 2 * (precision_u * recall_u) / (precision_u + recall_u) if (precision_u + recall_u) > 0 else 0
    
    # Supervised metrics
    cm_supervised = confusion_matrix(y_true, y_pred_supervised)
    tn_s, fp_s, fn_s, tp_s = cm_supervised.ravel()
    accuracy_s = (tp_s + tn_s) / (tp_s + tn_s + fp_s + fn_s)
    precision_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0
    recall_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
    f1_s = 2 * (precision_s * recall_s) / (precision_s + recall_s) if (precision_s + recall_s) > 0 else 0
    
    print(f"UNSUPERVISED (Isolation Forest):")
    print(f"  Accuracy:  {accuracy_u:.3f}")
    print(f"  Precision: {precision_u:.3f}")
    print(f"  Recall:    {recall_u:.3f}")
    print(f"  F1-Score:  {f1_u:.3f}")
    print(f"  True Positives: {tp_u}, False Positives: {fp_u}")
    print(f"  False Negatives: {fn_u}, True Negatives: {tn_u}")
    
    print(f"\nSUPERVISED (Random Forest):")
    print(f"  Accuracy:  {accuracy_s:.3f}")
    print(f"  Precision: {precision_s:.3f}")
    print(f"  Recall:    {recall_s:.3f}")
    print(f"  F1-Score:  {f1_s:.3f}")
    print(f"  True Positives: {tp_s}, False Positives: {fp_s}")
    print(f"  False Negatives: {fn_s}, True Negatives: {tn_s}")
    
    # Improvements
    print(f"\n📈 IMPROVEMENTS WITH SUPERVISED LEARNING:")
    print(f"  F1-Score:  {f1_s - f1_u:+.3f} ({((f1_s - f1_u) / f1_u * 100):+.1f}%)")
    print(f"  Accuracy:  {accuracy_s - accuracy_u:+.3f} ({((accuracy_s - accuracy_u) / accuracy_u * 100):+.1f}%)")
    print(f"  Precision: {precision_s - precision_u:+.3f} ({((precision_s - precision_u) / precision_u * 100):+.1f}%)")
    print(f"  Recall:    {recall_s - recall_u:+.3f} ({((recall_s - recall_u) / recall_u * 100):+.1f}%)")
    
    return {
        'unsupervised': {'accuracy': accuracy_u, 'precision': precision_u, 'recall': recall_u, 'f1': f1_u},
        'supervised': {'accuracy': accuracy_s, 'precision': precision_s, 'recall': recall_s, 'f1': f1_s}
    }

def show_detected_anomalies(df, predictions_unsupervised, predictions_supervised):
    """Show which anomalies were detected by each method"""
    print("\n🔍 ANOMALY DETECTION DETAILS")
    print("=" * 50)
    
    # Get true anomalies
    true_anomalies = df[df['is_anomaly'] == True].copy()
    
    print(f"📋 TRUE ANOMALIES IN DATA ({len(true_anomalies)} total):")
    for _, row in true_anomalies.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')} - {row['fund_id']}: "
              f"Rating={row['fund_rating']}, Value={row['fund_value']:,}, "
              f"Size={row['fund_size']:,}, Returns={row['fund_returns']:.3f}")
    
    # Check which were detected by unsupervised
    print(f"\n🔍 UNSUPERVISED DETECTION:")
    detected_unsupervised = true_anomalies[predictions_unsupervised['predictions'] == -1].copy()
    print(f"  Detected {len(detected_unsupervised)} out of {len(true_anomalies)} true anomalies")
    for _, row in detected_unsupervised.iterrows():
        print(f"  ✅ {row['date'].strftime('%Y-%m-%d')} - {row['fund_id']}")
    
    # Check which were detected by supervised
    print(f"\n🎯 SUPERVISED DETECTION:")
    detected_supervised = true_anomalies[predictions_supervised['predictions'] == -1].copy()
    print(f"  Detected {len(detected_supervised)} out of {len(true_anomalies)} true anomalies")
    for _, row in detected_supervised.iterrows():
        print(f"  ✅ {row['date'].strftime('%Y-%m-%d')} - {row['fund_id']}")
    
    # Show missed anomalies
    missed_unsupervised = true_anomalies[predictions_unsupervised['predictions'] == 1].copy()
    missed_supervised = true_anomalies[predictions_supervised['predictions'] == 1].copy()
    
    if len(missed_unsupervised) > 0:
        print(f"\n❌ UNSUPERVISED MISSED:")
        for _, row in missed_unsupervised.iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')} - {row['fund_id']}")
    
    if len(missed_supervised) > 0:
        print(f"\n❌ SUPERVISED MISSED:")
        for _, row in missed_supervised.iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')} - {row['fund_id']}")

def main():
    """Main demonstration"""
    print("🎯 SUPERVISED vs UNSUPERVISED ANOMALY DETECTION")
    print("Demonstrating the power of true labels in fund rating anomaly detection")
    print("=" * 80)
    
    # Create example data
    df = create_fund_example()
    print(f"Created {len(df)} fund records")
    print(f"True anomalies: {df['is_anomaly'].sum()}")
    
    # Test both approaches
    results = test_unsupervised_vs_supervised(df)
    
    # Show detailed results
    detector_unsupervised = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
    detector_unsupervised.train(df=df, business_key='fund_id', 
                               target_attributes=['fund_rating', 'fund_value', 'fund_size', 'fund_returns'],
                               time_column='date')
    predictions_unsupervised = detector_unsupervised.predict(df, business_key='fund_id', time_column='date')
    
    detector_supervised = GenericAnomalyDetector(algorithm='random_forest', contamination=0.1)
    detector_supervised.train(df=df, business_key='fund_id', 
                             target_attributes=['fund_rating', 'fund_value', 'fund_size', 'fund_returns'],
                             time_column='date', anomaly_labels='is_anomaly')
    predictions_supervised = detector_supervised.predict(df, business_key='fund_id', time_column='date')
    
    show_detected_anomalies(df, predictions_unsupervised, predictions_supervised)
    
    print("\n✅ DEMONSTRATION COMPLETED!")
    print("\n💡 KEY INSIGHTS:")
    print("   • Supervised learning with true labels provides significantly better accuracy")
    print("   • True labels help the model learn specific anomaly patterns")
    print("   • When you have labeled data, always use supervised learning")
    print("   • Unsupervised learning is still valuable when labels are not available")

if __name__ == "__main__":
    main()
