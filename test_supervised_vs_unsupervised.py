#!/usr/bin/env python3
"""
Supervised vs Unsupervised Anomaly Detection Comparison
Tests how much accuracy improves when true labels are provided
"""

import pandas as pd
import numpy as np
from generic_anomaly_detector import GenericAnomalyDetector
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def generate_labeled_data(n_entities=10, n_days=180, anomaly_rate=0.15):
    """Generate synthetic data with known anomaly labels"""
    print(f"Generating labeled data: {n_entities} entities, {n_days} days, {anomaly_rate*100}% anomaly rate")
    
    entities = [f'ENTITY_{i:03d}' for i in range(n_entities)]
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    all_data = []
    true_anomalies = set()
    
    for entity in entities:
        # Generate base patterns for each entity
        base_value = np.random.uniform(100, 1000)
        base_rating = np.random.uniform(2, 5)
        base_size = np.random.uniform(1000, 10000)
        
        for i, date in enumerate(dates):
            # Normal data with some randomness
            value = base_value + np.random.normal(0, base_value * 0.05)
            rating = base_rating + np.random.normal(0, 0.3)
            size = base_size + np.random.normal(0, base_size * 0.1)
            returns = np.random.normal(0.001, 0.02)
            
            # Inject anomalies
            is_anomaly = np.random.random() < anomaly_rate
            if is_anomaly:
                anomaly_type = np.random.choice(['spike', 'drop', 'rating_change', 'size_change'])
                
                if anomaly_type == 'spike':
                    value *= np.random.uniform(2, 5)  # 2-5x spike
                    rating = min(5, rating + np.random.uniform(1, 2))  # Rating spike
                elif anomaly_type == 'drop':
                    value *= np.random.uniform(0.1, 0.5)  # 50-90% drop
                    rating = max(1, rating - np.random.uniform(1, 2))  # Rating drop
                elif anomaly_type == 'rating_change':
                    rating = np.random.choice([1, 5])  # Extreme rating
                elif anomaly_type == 'size_change':
                    size *= np.random.uniform(0.1, 10)  # Extreme size change
                
                true_anomalies.add((entity, date))
            
            all_data.append({
                'date': date,
                'entity_id': entity,
                'value': max(0, value),
                'rating': max(1, min(5, rating)),
                'size': max(0, size),
                'returns': returns,
                'is_anomaly': is_anomaly
            })
    
    df = pd.DataFrame(all_data)
    return df, true_anomalies

def test_unsupervised_performance(df, true_anomalies):
    """Test unsupervised anomaly detection performance"""
    print("\n🔍 TESTING UNSUPERVISED PERFORMANCE")
    print("=" * 50)
    
    # Test different algorithms and contamination rates
    configs = [
        {'algorithm': 'isolation_forest', 'contamination': 0.1},
        {'algorithm': 'isolation_forest', 'contamination': 0.15},
        {'algorithm': 'isolation_forest', 'contamination': 0.2},
        {'algorithm': 'one_class_svm', 'contamination': 0.1},
        {'algorithm': 'one_class_svm', 'contamination': 0.15},
    ]
    
    best_unsupervised = None
    best_f1 = 0
    
    for config in configs:
        print(f"\nTesting {config['algorithm']} with contamination={config['contamination']}")
        
        try:
            # Train unsupervised model
            detector = GenericAnomalyDetector(
                algorithm=config['algorithm'], 
                contamination=config['contamination']
            )
            detector.train(
                df=df,
                business_key='entity_id',
                target_attributes=['value', 'rating', 'size', 'returns'],
                time_column='date'
            )
            
            # Make predictions
            predictions = detector.predict(df, business_key='entity_id', time_column='date')
            
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
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_unsupervised = {
                    'algorithm': config['algorithm'],
                    'contamination': config['contamination'],
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
                }
                
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    return best_unsupervised

def test_supervised_performance(df, true_anomalies):
    """Test supervised anomaly detection performance"""
    print("\n🎯 TESTING SUPERVISED PERFORMANCE")
    print("=" * 50)
    
    # Test different supervised algorithms
    configs = [
        {'algorithm': 'random_forest', 'contamination': 0.1},
        {'algorithm': 'random_forest', 'contamination': 0.15},
        {'algorithm': 'logistic_regression', 'contamination': 0.1},
        {'algorithm': 'logistic_regression', 'contamination': 0.15},
    ]
    
    best_supervised = None
    best_f1 = 0
    
    for config in configs:
        print(f"\nTesting {config['algorithm']} with contamination={config['contamination']}")
        
        try:
            # Train supervised model
            detector = GenericAnomalyDetector(
                algorithm=config['algorithm'], 
                contamination=config['contamination']
            )
            detector.train(
                df=df,
                business_key='entity_id',
                target_attributes=['value', 'rating', 'size', 'returns'],
                time_column='date',
                anomaly_labels='is_anomaly'  # Provide true labels
            )
            
            # Make predictions
            predictions = detector.predict(df, business_key='entity_id', time_column='date')
            
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
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_supervised = {
                    'algorithm': config['algorithm'],
                    'contamination': config['contamination'],
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
                }
                
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    return best_supervised

def compare_performance(unsupervised, supervised):
    """Compare supervised vs unsupervised performance"""
    print("\n" + "=" * 60)
    print("📊 SUPERVISED vs UNSUPERVISED COMPARISON")
    print("=" * 60)
    
    if not unsupervised or not supervised:
        print("❌ Could not complete comparison - missing results")
        return
    
    print(f"\n🏆 BEST UNSUPERVISED MODEL:")
    print(f"   Algorithm: {unsupervised['algorithm']}")
    print(f"   Contamination: {unsupervised['contamination']}")
    print(f"   F1-Score: {unsupervised['f1_score']:.3f}")
    print(f"   Accuracy: {unsupervised['accuracy']:.3f}")
    print(f"   Precision: {unsupervised['precision']:.3f}")
    print(f"   Recall: {unsupervised['recall']:.3f}")
    
    print(f"\n🎯 BEST SUPERVISED MODEL:")
    print(f"   Algorithm: {supervised['algorithm']}")
    print(f"   Contamination: {supervised['contamination']}")
    print(f"   F1-Score: {supervised['f1_score']:.3f}")
    print(f"   Accuracy: {supervised['accuracy']:.3f}")
    print(f"   Precision: {supervised['precision']:.3f}")
    print(f"   Recall: {supervised['recall']:.3f}")
    
    # Calculate improvements
    f1_improvement = supervised['f1_score'] - unsupervised['f1_score']
    accuracy_improvement = supervised['accuracy'] - unsupervised['accuracy']
    precision_improvement = supervised['precision'] - unsupervised['precision']
    recall_improvement = supervised['recall'] - unsupervised['recall']
    
    print(f"\n📈 PERFORMANCE IMPROVEMENTS:")
    print(f"   F1-Score: {f1_improvement:+.3f} ({f1_improvement/unsupervised['f1_score']*100:+.1f}%)")
    print(f"   Accuracy: {accuracy_improvement:+.3f} ({accuracy_improvement/unsupervised['accuracy']*100:+.1f}%)")
    print(f"   Precision: {precision_improvement:+.3f} ({precision_improvement/unsupervised['precision']*100:+.1f}%)")
    print(f"   Recall: {recall_improvement:+.3f} ({recall_improvement/unsupervised['recall']*100:+.1f}%)")
    
    # Calculate confusion matrix improvements
    print(f"\n🔍 CONFUSION MATRIX COMPARISON:")
    print(f"   Unsupervised - TP: {unsupervised['tp']}, FP: {unsupervised['fp']}, FN: {unsupervised['fn']}, TN: {unsupervised['tn']}")
    print(f"   Supervised   - TP: {supervised['tp']}, FP: {supervised['fp']}, FN: {supervised['fn']}, TN: {supervised['tn']}")
    
    tp_improvement = supervised['tp'] - unsupervised['tp']
    fp_improvement = supervised['fp'] - unsupervised['fp']
    fn_improvement = supervised['fn'] - unsupervised['fn']
    tn_improvement = supervised['tn'] - unsupervised['tn']
    
    print(f"   Improvements - TP: {tp_improvement:+d}, FP: {fp_improvement:+d}, FN: {fn_improvement:+d}, TN: {tn_improvement:+d}")
    
    # Summary
    print(f"\n✅ SUMMARY:")
    if f1_improvement > 0:
        print(f"   🎯 Supervised learning provides {f1_improvement:.3f} better F1-Score")
        print(f"   📈 That's a {f1_improvement/unsupervised['f1_score']*100:.1f}% improvement!")
    else:
        print(f"   ⚠️  Supervised learning didn't improve F1-Score in this test")
    
    if accuracy_improvement > 0:
        print(f"   🎯 Supervised learning provides {accuracy_improvement:.3f} better accuracy")
        print(f"   📈 That's a {accuracy_improvement/unsupervised['accuracy']*100:.1f}% improvement!")
    
    if recall_improvement > 0:
        print(f"   🎯 Supervised learning catches {recall_improvement:.3f} more anomalies")
        print(f"   📈 That's a {recall_improvement/unsupervised['recall']*100:.1f}% improvement!")

def main():
    """Main comparison test"""
    print("🎯 Supervised vs Unsupervised Anomaly Detection Comparison")
    print("=" * 70)
    
    # Generate test data
    df, true_anomalies = generate_labeled_data(n_entities=8, n_days=150, anomaly_rate=0.12)
    print(f"Generated {len(df)} records with {len(true_anomalies)} true anomalies")
    
    # Test unsupervised performance
    best_unsupervised = test_unsupervised_performance(df, true_anomalies)
    
    # Test supervised performance
    best_supervised = test_supervised_performance(df, true_anomalies)
    
    # Compare results
    compare_performance(best_unsupervised, best_supervised)
    
    print("\n✅ Comparison completed!")
    print("\n💡 KEY TAKEAWAYS:")
    print("   • Supervised learning typically provides 20-50% better performance")
    print("   • True labels help the model learn specific anomaly patterns")
    print("   • Use supervised when you have labeled data available")
    print("   • Use unsupervised when labels are not available or expensive to obtain")

if __name__ == "__main__":
    main()
