#!/usr/bin/env python3
"""
Fund Rating Anomaly Detection Accuracy Test
Tests the generic engine specifically on fund rating data
"""

import pandas as pd
import numpy as np
from generic_anomaly_detector import GenericAnomalyDetector
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def generate_fund_data(n_funds=20, n_days=365, anomaly_rate=0.1):
    """Generate realistic fund data with known anomalies"""
    print(f"Generating fund data: {n_funds} funds, {n_days} days, {anomaly_rate*100}% anomaly rate")
    
    funds = [f'FUND_{i:03d}' for i in range(n_funds)]
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    all_data = []
    true_anomalies = set()
    
    for fund in funds:
        # Each fund has different characteristics
        base_rating = np.random.uniform(2.5, 4.5)
        base_value = np.random.uniform(100000, 5000000)
        base_size = np.random.uniform(1000, 50000)
        volatility = np.random.uniform(0.1, 0.3)
        
        for i, date in enumerate(dates):
            # Normal fund behavior
            value = base_value * (1 + np.random.normal(0, volatility * 0.1))
            rating = base_rating + np.random.normal(0, 0.2)
            size = base_size * (1 + np.random.normal(0, 0.05))
            returns = np.random.normal(0.0005, volatility)
            
            # Inject fund-specific anomalies
            is_anomaly = np.random.random() < anomaly_rate
            if is_anomaly:
                anomaly_type = np.random.choice([
                    'rating_spike', 'rating_drop', 'value_crash', 
                    'value_surge', 'size_explosion', 'extreme_returns'
                ])
                
                if anomaly_type == 'rating_spike':
                    rating = min(5, rating + np.random.uniform(1.5, 2.5))
                elif anomaly_type == 'rating_drop':
                    rating = max(1, rating - np.random.uniform(1.5, 2.5))
                elif anomaly_type == 'value_crash':
                    value *= np.random.uniform(0.1, 0.4)
                    rating = max(1, rating - 1)
                elif anomaly_type == 'value_surge':
                    value *= np.random.uniform(2, 5)
                    rating = min(5, rating + 1)
                elif anomaly_type == 'size_explosion':
                    size *= np.random.uniform(5, 20)
                elif anomaly_type == 'extreme_returns':
                    returns = np.random.choice([-0.5, 0.5])  # ±50% returns
                
                true_anomalies.add((fund, date))
            
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
    return df, true_anomalies

def test_fund_anomaly_detection():
    """Test fund anomaly detection accuracy"""
    print("🎯 Fund Rating Anomaly Detection Accuracy Test")
    print("=" * 60)
    
    # Generate test data
    df, true_anomalies = generate_fund_data(n_funds=15, n_days=180, anomaly_rate=0.12)
    print(f"Generated {len(df)} fund records with {len(true_anomalies)} true anomalies")
    
    # Test different configurations
    configs = [
        {'algorithm': 'isolation_forest', 'contamination': 0.1},
        {'algorithm': 'isolation_forest', 'contamination': 0.15},
        {'algorithm': 'isolation_forest', 'contamination': 0.2},
        {'algorithm': 'one_class_svm', 'contamination': 0.1},
        {'algorithm': 'one_class_svm', 'contamination': 0.15},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['algorithm']} with contamination={config['contamination']}")
        
        try:
            # Train model
            detector = GenericAnomalyDetector(
                algorithm=config['algorithm'], 
                contamination=config['contamination']
            )
            detector.train(
                df=df,
                business_key='fund_id',
                target_attributes=['fund_rating', 'fund_value', 'fund_size', 'fund_returns'],
                time_column='date'
            )
            
            # Make predictions
            predictions = detector.predict(df, business_key='fund_id', time_column='date')
            
            # Calculate metrics
            y_true = df['is_anomaly'].astype(int)
            y_pred = predictions['predictions']
            
            # Calculate detailed metrics
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate anomaly detection rate
            detected_anomalies = set()
            for i, (fund, date) in enumerate(zip(df['fund_id'], df['date'])):
                if y_pred[i] == 1:
                    detected_anomalies.add((fund, date))
            
            anomaly_detection_rate = len(detected_anomalies.intersection(true_anomalies)) / len(true_anomalies) if len(true_anomalies) > 0 else 0
            false_positive_rate = len(detected_anomalies - true_anomalies) / len(detected_anomalies) if len(detected_anomalies) > 0 else 0
            
            result = {
                'algorithm': config['algorithm'],
                'contamination': config['contamination'],
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'anomaly_detection_rate': anomaly_detection_rate,
                'false_positive_rate': false_positive_rate,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            }
            
            results.append(result)
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
            print(f"  Anomaly Detection Rate: {anomaly_detection_rate:.3f}")
            print(f"  False Positive Rate: {false_positive_rate:.3f}")
            print(f"  True Positives: {tp}, False Positives: {fp}")
            print(f"  False Negatives: {fn}, True Negatives: {tn}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    # Analyze results
    if results:
        print("\n" + "=" * 60)
        print("📊 FUND ANOMALY DETECTION RESULTS")
        print("=" * 60)
        
        df_results = pd.DataFrame(results)
        
        # Best model
        best_model = df_results.loc[df_results['f1_score'].idxmax()]
        print(f"\n🏆 BEST MODEL FOR FUND DATA:")
        print(f"   Algorithm: {best_model['algorithm']}")
        print(f"   Contamination: {best_model['contamination']}")
        print(f"   F1-Score: {best_model['f1_score']:.3f}")
        print(f"   Accuracy: {best_model['accuracy']:.3f}")
        print(f"   Anomaly Detection Rate: {best_model['anomaly_detection_rate']:.3f}")
        
        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   Average Accuracy: {df_results['accuracy'].mean():.3f}")
        print(f"   Average F1-Score: {df_results['f1_score'].mean():.3f}")
        print(f"   Average Anomaly Detection Rate: {df_results['anomaly_detection_rate'].mean():.3f}")
        print(f"   Average False Positive Rate: {df_results['false_positive_rate'].mean():.3f}")
        
        # Algorithm comparison
        print(f"\n🔍 ALGORITHM COMPARISON:")
        algo_summary = df_results.groupby('algorithm').agg({
            'accuracy': 'mean',
            'f1_score': 'mean',
            'anomaly_detection_rate': 'mean',
            'false_positive_rate': 'mean'
        }).round(3)
        print(algo_summary)
        
        # Show detailed confusion matrix for best model
        print(f"\n📋 DETAILED ANALYSIS FOR BEST MODEL:")
        best_config = {'algorithm': best_model['algorithm'], 'contamination': best_model['contamination']}
        
        # Re-run best model for detailed analysis
        detector = GenericAnomalyDetector(
            algorithm=best_config['algorithm'], 
            contamination=best_config['contamination']
        )
        detector.train(
            df=df,
            business_key='fund_id',
            target_attributes=['fund_rating', 'fund_value', 'fund_size', 'fund_returns'],
            time_column='date'
        )
        predictions = detector.predict(df, business_key='fund_id', time_column='date')
        
        y_true = df['is_anomaly'].astype(int)
        y_pred = predictions['predictions']
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        # Show some example anomalies
        print(f"\n🔍 EXAMPLE DETECTED ANOMALIES:")
        anomaly_examples = df[y_pred == 1].head(10)
        for _, row in anomaly_examples.iterrows():
            status = "✅ TRUE ANOMALY" if row['is_anomaly'] else "❌ FALSE POSITIVE"
            print(f"  {status} - Fund: {row['fund_id']}, Date: {row['date'].strftime('%Y-%m-%d')}, "
                  f"Rating: {row['fund_rating']}, Value: {row['fund_value']:,}")

if __name__ == "__main__":
    test_fund_anomaly_detection()
