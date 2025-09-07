#!/usr/bin/env python3
"""
Comprehensive Accuracy Test for Generic Anomaly Detection Engine
Tests the model's ability to detect known anomalies across different scenarios
"""

import pandas as pd
import numpy as np
from generic_anomaly_detector import GenericAnomalyDetector
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AnomalyAccuracyTester:
    def __init__(self):
        self.results = {}
        
    def generate_synthetic_data(self, n_entities=5, n_days=365, anomaly_rate=0.1):
        """Generate synthetic data with known anomalies"""
        print(f"Generating synthetic data: {n_entities} entities, {n_days} days, {anomaly_rate*100}% anomaly rate")
        
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
    
    def test_algorithm_accuracy(self, algorithm, df, true_anomalies, contamination=0.1):
        """Test accuracy of a specific algorithm"""
        print(f"\nTesting {algorithm} with contamination={contamination}")
        
        try:
            # Train model
            detector = GenericAnomalyDetector(algorithm=algorithm, contamination=contamination)
            detector.train(
                df=df,
                business_key='entity_id',
                target_attributes=['value', 'rating', 'size', 'returns'],
                time_column='date'
            )
            
            # Make predictions
            predictions = detector.predict(df, business_key='entity_id', time_column='date')
            
            # Calculate accuracy metrics
            y_true = df['is_anomaly'].astype(int)
            y_pred = predictions['predictions']
            
            # Calculate metrics
            accuracy = (y_true == y_pred).mean()
            precision = confusion_matrix(y_true, y_pred)[1, 1] / (confusion_matrix(y_true, y_pred)[1, 1] + confusion_matrix(y_true, y_pred)[0, 1]) if (confusion_matrix(y_true, y_pred)[1, 1] + confusion_matrix(y_true, y_pred)[0, 1]) > 0 else 0
            recall = confusion_matrix(y_true, y_pred)[1, 1] / (confusion_matrix(y_true, y_pred)[1, 1] + confusion_matrix(y_true, y_pred)[1, 0]) if (confusion_matrix(y_true, y_pred)[1, 1] + confusion_matrix(y_true, y_pred)[1, 0]) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate AUC if possible
            try:
                auc = roc_auc_score(y_true, y_pred)
            except:
                auc = 0.5
            
            # Calculate anomaly detection rate
            detected_anomalies = set()
            for i, (entity, date) in enumerate(zip(df['entity_id'], df['date'])):
                if y_pred[i] == 1:
                    detected_anomalies.add((entity, date))
            
            anomaly_detection_rate = len(detected_anomalies.intersection(true_anomalies)) / len(true_anomalies) if len(true_anomalies) > 0 else 0
            false_positive_rate = len(detected_anomalies - true_anomalies) / len(detected_anomalies) if len(detected_anomalies) > 0 else 0
            
            results = {
                'algorithm': algorithm,
                'contamination': contamination,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'anomaly_detection_rate': anomaly_detection_rate,
                'false_positive_rate': false_positive_rate,
                'total_anomalies': len(true_anomalies),
                'detected_anomalies': len(detected_anomalies),
                'true_positives': len(detected_anomalies.intersection(true_anomalies)),
                'false_positives': len(detected_anomalies - true_anomalies),
                'false_negatives': len(true_anomalies - detected_anomalies)
            }
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
            print(f"  AUC: {auc:.3f}")
            print(f"  Anomaly Detection Rate: {anomaly_detection_rate:.3f}")
            print(f"  False Positive Rate: {false_positive_rate:.3f}")
            
            return results
            
        except Exception as e:
            print(f"  Error testing {algorithm}: {str(e)}")
            return None
    
    def run_comprehensive_test(self):
        """Run comprehensive accuracy tests"""
        print("🚀 Starting Comprehensive Anomaly Detection Accuracy Test")
        print("=" * 60)
        
        # Generate test data
        df, true_anomalies = self.generate_synthetic_data(n_entities=10, n_days=180, anomaly_rate=0.15)
        print(f"Generated {len(df)} records with {len(true_anomalies)} true anomalies")
        
        # Test different algorithms
        algorithms = ['isolation_forest', 'one_class_svm', 'local_outlier_factor']
        contaminations = [0.05, 0.1, 0.15, 0.2]
        
        all_results = []
        
        for algorithm in algorithms:
            for contamination in contaminations:
                result = self.test_algorithm_accuracy(algorithm, df, true_anomalies, contamination)
                if result:
                    all_results.append(result)
        
        # Analyze results
        self.analyze_results(all_results)
        
        return all_results
    
    def analyze_results(self, results):
        """Analyze and visualize results"""
        if not results:
            print("No results to analyze")
            return
        
        print("\n" + "=" * 60)
        print("📊 ACCURACY ANALYSIS RESULTS")
        print("=" * 60)
        
        # Convert to DataFrame for analysis
        df_results = pd.DataFrame(results)
        
        # Best performing model
        best_model = df_results.loc[df_results['f1_score'].idxmax()]
        print(f"\n🏆 BEST PERFORMING MODEL:")
        print(f"   Algorithm: {best_model['algorithm']}")
        print(f"   Contamination: {best_model['contamination']}")
        print(f"   F1-Score: {best_model['f1_score']:.3f}")
        print(f"   Accuracy: {best_model['accuracy']:.3f}")
        print(f"   Anomaly Detection Rate: {best_model['anomaly_detection_rate']:.3f}")
        
        # Algorithm comparison
        print(f"\n📈 ALGORITHM COMPARISON:")
        algo_summary = df_results.groupby('algorithm').agg({
            'accuracy': 'mean',
            'f1_score': 'mean',
            'anomaly_detection_rate': 'mean',
            'false_positive_rate': 'mean'
        }).round(3)
        print(algo_summary)
        
        # Contamination analysis
        print(f"\n🎯 CONTAMINATION ANALYSIS:")
        cont_summary = df_results.groupby('contamination').agg({
            'accuracy': 'mean',
            'f1_score': 'mean',
            'anomaly_detection_rate': 'mean',
            'false_positive_rate': 'mean'
        }).round(3)
        print(cont_summary)
        
        # Create visualizations
        self.create_visualizations(df_results)
    
    def create_visualizations(self, df_results):
        """Create visualization plots"""
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Anomaly Detection Accuracy Analysis', fontsize=16, fontweight='bold')
            
            # 1. Algorithm Performance Comparison
            algo_metrics = df_results.groupby('algorithm')[['accuracy', 'f1_score', 'anomaly_detection_rate']].mean()
            algo_metrics.plot(kind='bar', ax=axes[0, 0], color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[0, 0].set_title('Algorithm Performance Comparison')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].legend()
            
            # 2. Contamination vs Performance
            cont_metrics = df_results.groupby('contamination')[['accuracy', 'f1_score']].mean()
            cont_metrics.plot(kind='line', ax=axes[0, 1], marker='o', linewidth=2)
            axes[0, 1].set_title('Contamination vs Performance')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_xlabel('Contamination Rate')
            axes[0, 1].legend()
            
            # 3. Precision vs Recall Scatter
            scatter = axes[1, 0].scatter(df_results['precision'], df_results['recall'], 
                                       c=df_results['f1_score'], cmap='viridis', s=100, alpha=0.7)
            axes[1, 0].set_title('Precision vs Recall (Color = F1-Score)')
            axes[1, 0].set_xlabel('Precision')
            axes[1, 0].set_ylabel('Recall')
            plt.colorbar(scatter, ax=axes[1, 0])
            
            # 4. Anomaly Detection Rate vs False Positive Rate
            axes[1, 1].scatter(df_results['false_positive_rate'], df_results['anomaly_detection_rate'], 
                              c=df_results['f1_score'], cmap='plasma', s=100, alpha=0.7)
            axes[1, 1].set_title('Anomaly Detection vs False Positive Rate')
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('Anomaly Detection Rate')
            
            plt.tight_layout()
            plt.savefig('anomaly_detection_accuracy_analysis.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 Visualizations saved as 'anomaly_detection_accuracy_analysis.png'")
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
    
    def test_edge_cases(self):
        """Test edge cases and robustness"""
        print("\n🔍 TESTING EDGE CASES")
        print("=" * 40)
        
        # Test 1: Very low anomaly rate
        print("\n1. Testing very low anomaly rate (1%)...")
        df_low, true_anomalies_low = self.generate_synthetic_data(n_entities=5, n_days=100, anomaly_rate=0.01)
        result_low = self.test_algorithm_accuracy('isolation_forest', df_low, true_anomalies_low, contamination=0.05)
        
        # Test 2: Very high anomaly rate
        print("\n2. Testing very high anomaly rate (30%)...")
        df_high, true_anomalies_high = self.generate_synthetic_data(n_entities=5, n_days=100, anomaly_rate=0.3)
        result_high = self.test_algorithm_accuracy('isolation_forest', df_high, true_anomalies_high, contamination=0.25)
        
        # Test 3: Single entity
        print("\n3. Testing single entity...")
        df_single = df_low[df_low['entity_id'] == 'ENTITY_000'].copy()
        true_anomalies_single = {(e, d) for e, d in true_anomalies_low if e == 'ENTITY_000'}
        result_single = self.test_algorithm_accuracy('isolation_forest', df_single, true_anomalies_single, contamination=0.1)
        
        print(f"\nEdge Case Results:")
        print(f"  Low anomaly rate (1%): F1={result_low['f1_score']:.3f}" if result_low else "  Low anomaly rate: Failed")
        print(f"  High anomaly rate (30%): F1={result_high['f1_score']:.3f}" if result_high else "  High anomaly rate: Failed")
        print(f"  Single entity: F1={result_single['f1_score']:.3f}" if result_single else "  Single entity: Failed")

def main():
    """Main test execution"""
    print("🎯 Generic Anomaly Detection Engine - Accuracy Test")
    print("=" * 60)
    
    tester = AnomalyAccuracyTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Test edge cases
    tester.test_edge_cases()
    
    print("\n✅ Accuracy testing completed!")
    print("Check 'anomaly_detection_accuracy_analysis.png' for detailed visualizations")

if __name__ == "__main__":
    main()
