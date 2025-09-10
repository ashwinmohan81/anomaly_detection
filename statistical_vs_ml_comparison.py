"""
Comprehensive Comparison: Statistical vs ML-based Anomaly Detection

This script compares the performance of:
1. Statistical Anomaly Detection (our new approach)
2. ML-based Anomaly Detection (Isolation Forest)
3. Hybrid Approach (combining both methods)

The comparison includes:
- Accuracy metrics
- Interpretability analysis
- Performance characteristics
- Use case recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from statistical_anomaly_detector import StatisticalAnomalyDetector
from generic_anomaly_detector import GenericAnomalyDetector
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetectionComparison:
    """
    Comprehensive comparison of different anomaly detection approaches.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize comparison with data.
        
        Args:
            df: Input DataFrame with time series data
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Split data for training and testing
        self.train_data = self.df.sample(frac=0.7, random_state=42)
        self.test_data = self.df.drop(self.train_data.index)
        
        print(f"📊 Data Split:")
        print(f"  Training: {len(self.train_data):,} records")
        print(f"  Testing: {len(self.test_data):,} records")
        
    def create_ground_truth(self, method: str = 'statistical') -> pd.DataFrame:
        """
        Create ground truth labels for evaluation.
        
        Args:
            method: Method to use for ground truth ('statistical', 'ml', 'hybrid')
            
        Returns:
            DataFrame with ground truth labels
        """
        print(f"🏷️ Creating ground truth using {method} method...")
        
        if method == 'statistical':
            # Use statistical method with conservative thresholds
            detector = StatisticalAnomalyDetector(
                z_threshold=2.5,  # More conservative
                momentum_threshold=0.15,
                percentile_threshold=0.05,
                volatility_threshold=2.5
            )
            
            detector.train(
                df=self.train_data,
                business_key='index_id',
                target_attributes=['index_return', 'num_constituents'],
                time_column='date'
            )
            
            predictions = detector.predict(self.test_data, 'index_id', 'date')
            ground_truth = predictions['predictions']
            
        elif method == 'ml':
            # Use ML method
            detector = GenericAnomalyDetector(
                algorithm='isolation_forest',
                contamination=0.1
            )
            
            detector.train(
                df=self.train_data,
                business_key='index_id',
                target_attributes=['index_return', 'num_constituents'],
                time_column='date'
            )
            
            predictions = detector.predict(self.test_data, 'index_id', 'date')
            ground_truth = predictions['predictions']
            
        else:  # hybrid
            # Combine both methods
            stat_detector = StatisticalAnomalyDetector(
                z_threshold=2.0,
                momentum_threshold=0.1,
                percentile_threshold=0.1,
                volatility_threshold=2.0
            )
            
            ml_detector = GenericAnomalyDetector(
                algorithm='isolation_forest',
                contamination=0.1
            )
            
            stat_detector.train(
                df=self.train_data,
                business_key='index_id',
                target_attributes=['index_return', 'num_constituents'],
                time_column='date'
            )
            
            ml_detector.train(
                df=self.train_data,
                business_key='index_id',
                target_attributes=['index_return', 'num_constituents'],
                time_column='date'
            )
            
            stat_pred = stat_detector.predict(self.test_data, 'index_id', 'date')
            ml_pred = ml_detector.predict(self.test_data, 'index_id', 'date')
            
            # Hybrid: anomaly if either method detects it
            ground_truth = [a or b for a, b in zip(stat_pred['predictions'], ml_pred['predictions'])]
        
        return ground_truth
    
    def evaluate_statistical_method(self, ground_truth: List[bool]) -> Dict[str, Any]:
        """
        Evaluate statistical anomaly detection method.
        
        Args:
            ground_truth: Ground truth labels
            
        Returns:
            Evaluation results
        """
        print("📊 Evaluating Statistical Method...")
        
        # Train statistical detector
        detector = StatisticalAnomalyDetector(
            z_threshold=2.0,
            momentum_threshold=0.1,
            percentile_threshold=0.1,
            volatility_threshold=2.0
        )
        
        detector.train(
            df=self.train_data,
            business_key='index_id',
            target_attributes=['index_return', 'num_constituents'],
            time_column='date'
        )
        
        # Predict on test data
        predictions = detector.predict(self.test_data, 'index_id', 'date')
        pred_labels = predictions['predictions']
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth, pred_labels)
        precision = precision_score(ground_truth, pred_labels, zero_division=0)
        recall = recall_score(ground_truth, pred_labels, zero_division=0)
        f1 = f1_score(ground_truth, pred_labels, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, pred_labels)
        
        # Anomaly breakdown
        anomaly_breakdown = {}
        for attr in ['index_return', 'num_constituents']:
            anomaly_breakdown[attr] = {
                'z_score': sum(self.test_data[f'{attr}_z_score'].abs() > 2.0) if f'{attr}_z_score' in self.test_data.columns else 0,
                'volatility': sum(self.test_data[f'{attr}_volatility_ratio'] > 2.0) if f'{attr}_volatility_ratio' in self.test_data.columns else 0,
                'momentum': sum(self.test_data[f'{attr}_momentum_5'].abs() > 0.1) if f'{attr}_momentum_5' in self.test_data.columns else 0,
                'percentile': sum((self.test_data[f'{attr}_percentile_20'] < 0.1) | (self.test_data[f'{attr}_percentile_20'] > 0.9)) if f'{attr}_percentile_20' in self.test_data.columns else 0
            }
        
        return {
            'method': 'Statistical',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'anomaly_rate': sum(pred_labels) / len(pred_labels),
            'anomaly_breakdown': anomaly_breakdown,
            'interpretability': 'High',
            'explainability': 'Full - each anomaly has clear statistical reasoning',
            'performance': 'Fast - O(n) complexity',
            'robustness': 'High - based on statistical principles'
        }
    
    def evaluate_ml_method(self, ground_truth: List[bool]) -> Dict[str, Any]:
        """
        Evaluate ML-based anomaly detection method.
        
        Args:
            ground_truth: Ground truth labels
            
        Returns:
            Evaluation results
        """
        print("🤖 Evaluating ML Method...")
        
        # Train ML detector
        detector = GenericAnomalyDetector(
            algorithm='isolation_forest',
            contamination=0.1
        )
        
        detector.train(
            df=self.train_data,
            business_key='index_id',
            target_attributes=['index_return', 'num_constituents'],
            time_column='date'
        )
        
        # Predict on test data
        predictions = detector.predict(self.test_data, 'index_id', 'date')
        pred_labels = predictions['predictions']
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth, pred_labels)
        precision = precision_score(ground_truth, pred_labels, zero_division=0)
        recall = recall_score(ground_truth, pred_labels, zero_division=0)
        f1 = f1_score(ground_truth, pred_labels, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, pred_labels)
        
        return {
            'method': 'ML (Isolation Forest)',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'anomaly_rate': sum(pred_labels) / len(pred_labels),
            'interpretability': 'Low',
            'explainability': 'Limited - black box approach',
            'performance': 'Medium - O(n log n) complexity',
            'robustness': 'Medium - depends on data quality'
        }
    
    def evaluate_hybrid_method(self, ground_truth: List[bool]) -> Dict[str, Any]:
        """
        Evaluate hybrid anomaly detection method.
        
        Args:
            ground_truth: Ground truth labels
            
        Returns:
            Evaluation results
        """
        print("🔀 Evaluating Hybrid Method...")
        
        # Train both detectors
        stat_detector = StatisticalAnomalyDetector(
            z_threshold=2.0,
            momentum_threshold=0.1,
            percentile_threshold=0.1,
            volatility_threshold=2.0
        )
        
        ml_detector = GenericAnomalyDetector(
            algorithm='isolation_forest',
            contamination=0.1
        )
        
        stat_detector.train(
            df=self.train_data,
            business_key='index_id',
            target_attributes=['index_return', 'num_constituents'],
            time_column='date'
        )
        
        ml_detector.train(
            df=self.train_data,
            business_key='index_id',
            target_attributes=['index_return', 'num_constituents'],
            time_column='date'
        )
        
        # Predict with both methods
        stat_pred = stat_detector.predict(self.test_data, 'index_id', 'date')
        ml_pred = ml_detector.predict(self.test_data, 'index_id', 'date')
        
        # Hybrid: anomaly if either method detects it
        pred_labels = [a or b for a, b in zip(stat_pred['predictions'], ml_pred['predictions'])]
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth, pred_labels)
        precision = precision_score(ground_truth, pred_labels, zero_division=0)
        recall = recall_score(ground_truth, pred_labels, zero_division=0)
        f1 = f1_score(ground_truth, pred_labels, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, pred_labels)
        
        return {
            'method': 'Hybrid (Statistical + ML)',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'anomaly_rate': sum(pred_labels) / len(pred_labels),
            'interpretability': 'Medium',
            'explainability': 'Partial - statistical reasoning + ML patterns',
            'performance': 'Medium - combines both approaches',
            'robustness': 'High - leverages strengths of both methods'
        }
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """
        Run comprehensive comparison of all methods.
        
        Returns:
            Complete comparison results
        """
        print("🚀 Running Comprehensive Anomaly Detection Comparison")
        print("=" * 70)
        
        # Create ground truth using statistical method (most interpretable)
        ground_truth = self.create_ground_truth('statistical')
        
        # Evaluate all methods
        results = {}
        results['statistical'] = self.evaluate_statistical_method(ground_truth)
        results['ml'] = self.evaluate_ml_method(ground_truth)
        results['hybrid'] = self.evaluate_hybrid_method(ground_truth)
        
        return results
    
    def print_comparison_results(self, results: Dict[str, Any]):
        """
        Print comprehensive comparison results.
        
        Args:
            results: Comparison results dictionary
        """
        print("\\n📊 COMPREHENSIVE ANOMALY DETECTION COMPARISON")
        print("=" * 70)
        
        # Performance metrics comparison
        print("\\n🎯 Performance Metrics Comparison:")
        print("-" * 50)
        print(f"{'Method':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Anomaly Rate':<12}")
        print("-" * 50)
        
        for method, result in results.items():
            print(f"{result['method']:<20} {result['accuracy']:<10.3f} {result['precision']:<10.3f} "
                  f"{result['recall']:<10.3f} {result['f1_score']:<10.3f} {result['anomaly_rate']:<12.3f}")
        
        # Qualitative comparison
        print("\\n🔍 Qualitative Comparison:")
        print("-" * 50)
        print(f"{'Method':<20} {'Interpretability':<15} {'Explainability':<20} {'Performance':<12} {'Robustness':<10}")
        print("-" * 50)
        
        for method, result in results.items():
            print(f"{result['method']:<20} {result['interpretability']:<15} {result['explainability']:<20} "
                  f"{result['performance']:<12} {result['robustness']:<10}")
        
        # Confusion matrices
        print("\\n📈 Confusion Matrices:")
        print("-" * 50)
        
        for method, result in results.items():
            print(f"\\n{result['method']}:")
            cm = result['confusion_matrix']
            print(f"  True Negatives: {cm[0,0]:,}")
            print(f"  False Positives: {cm[0,1]:,}")
            print(f"  False Negatives: {cm[1,0]:,}")
            print(f"  True Positives: {cm[1,1]:,}")
        
        # Recommendations
        print("\\n💡 Recommendations:")
        print("-" * 50)
        
        # Find best method by F1-score
        best_method = max(results.items(), key=lambda x: x[1]['f1_score'])
        print(f"🏆 Best Overall Performance: {best_method[1]['method']} (F1-Score: {best_method[1]['f1_score']:.3f})")
        
        # Find most interpretable
        interpretable_methods = [m for m, r in results.items() if r['interpretability'] == 'High']
        if interpretable_methods:
            print(f"🔍 Most Interpretable: {results[interpretable_methods[0]]['method']}")
        
        # Use case recommendations
        print("\\n📋 Use Case Recommendations:")
        print("  📊 Statistical Method:")
        print("    - Best for: Financial analysis, regulatory reporting, explainable AI")
        print("    - When: Interpretability is critical, regulatory compliance required")
        print("    - Pros: Fully explainable, fast, robust")
        print("    - Cons: May miss complex patterns")
        
        print("\\n  🤖 ML Method:")
        print("    - Best for: Complex pattern detection, large datasets")
        print("    - When: Accuracy is more important than interpretability")
        print("    - Pros: Can detect complex patterns, good for large datasets")
        print("    - Cons: Black box, harder to explain")
        
        print("\\n  🔀 Hybrid Method:")
        print("    - Best for: Production systems, balanced approach")
        print("    - When: Need both accuracy and some interpretability")
        print("    - Pros: Combines strengths of both approaches")
        print("    - Cons: More complex, higher computational cost")
        
        # Target achievement
        print("\\n🎯 Target Achievement Analysis:")
        print("-" * 50)
        target_rate = 0.6
        
        for method, result in results.items():
            achieved = result['anomaly_rate'] >= target_rate
            status = "✅ ACHIEVED" if achieved else "❌ NOT MET"
            print(f"  {result['method']}: {result['anomaly_rate']*100:.1f}% >= {target_rate*100:.0f}% = {status}")


def main():
    """Main function to run the comparison."""
    print("🚀 Starting Comprehensive Anomaly Detection Comparison")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv('index_anomaly_data.csv')
    
    # Create comparison instance
    comparison = AnomalyDetectionComparison(df)
    
    # Run comprehensive comparison
    results = comparison.run_comprehensive_comparison()
    
    # Print results
    comparison.print_comparison_results(results)
    
    print("\\n✅ Comprehensive comparison complete!")


if __name__ == "__main__":
    main()
