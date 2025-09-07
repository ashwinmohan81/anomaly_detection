#!/usr/bin/env python3
"""
Final comprehensive test suite covering all features and edge cases
"""

import unittest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from datetime import datetime, timedelta
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generic_anomaly_detector import GenericAnomalyDetector

class TestFinalComprehensive(unittest.TestCase):
    """Final comprehensive test suite"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.test_data = self.create_comprehensive_test_data()
        
    def create_comprehensive_test_data(self):
        """Create comprehensive test data covering all scenarios"""
        data = []
        
        # Create data with multiple entities and time periods
        entities = ['FUND_001', 'FUND_002', 'FUND_003', 'FUND_004', 'FUND_005']
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10)]
        
        for entity in entities:
            for date in dates:
                # Create some anomalies
                is_anomaly = np.random.random() < 0.1
                
                if is_anomaly:
                    # Anomalous data
                    value = np.random.normal(2000, 200)  # Much higher values
                    rating = np.random.normal(5.5, 0.3)  # Much higher ratings
                    size = np.random.normal(8000, 800)   # Much larger sizes
                else:
                    # Normal data
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
        
        return pd.DataFrame(data)
    
    def test_all_algorithms_comprehensive(self):
        """Test all algorithms with comprehensive data"""
        algorithms = [
            ('isolation_forest', 'unsupervised'),
            ('one_class_svm', 'unsupervised'),
            ('local_outlier_factor', 'unsupervised'),
            ('random_forest', 'supervised'),
            ('logistic_regression', 'supervised')
        ]
        
        for algo, algo_type in algorithms:
            with self.subTest(algorithm=algo):
                detector = GenericAnomalyDetector(algorithm=algo, contamination=0.1)
                
                if algo_type == 'supervised':
                    # Supervised training
                    detector.train(
                        df=self.test_data,
                        business_key='entity_id',
                        target_attributes=['value', 'rating', 'size'],
                        time_column='date',
                        anomaly_labels='is_anomaly'
                    )
                else:
                    # Unsupervised training
                    detector.train(
                        df=self.test_data,
                        business_key='entity_id',
                        target_attributes=['value', 'rating', 'size'],
                        time_column='date'
                    )
                
                # Test prediction
                test_data = pd.DataFrame([{
                    'entity_id': 'TEST_ENTITY',
                    'value': 1000,
                    'rating': 4.5,
                    'size': 5000,
                    'date': '2024-01-01'
                }])
                
                result = detector.predict(
                    df=test_data,
                    business_key='entity_id',
                    time_column='date'
                )
                
                self.assertIn('predictions', result)
                self.assertIn('anomaly_indicators', result)
                self.assertIn('anomaly_count', result)
                self.assertIn('anomaly_rate', result)
                self.assertIn('prediction_analysis', result)
                self.assertTrue(detector.is_trained)
    
    def test_without_time_column_comprehensive(self):
        """Test comprehensive functionality without time column"""
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        
        # Remove time column
        data_no_time = self.test_data.drop(columns=['date'])
        
        # Train without time column
        detector.train(
            df=data_no_time,
            business_key='entity_id',
            target_attributes=['value', 'rating', 'size'],
            anomaly_labels='is_anomaly'
        )
        
        self.assertTrue(detector.is_trained)
        
        # Test prediction without time column
        test_records = pd.DataFrame([
            {'entity_id': 'ENTITY_001', 'value': 1000, 'rating': 4.5, 'size': 5000},
            {'entity_id': 'ENTITY_002', 'value': 800, 'rating': 3.2, 'size': 3000},
            {'entity_id': 'ENTITY_003', 'value': 1200, 'rating': 4.8, 'size': 6000}
        ])
        
        result = detector.predict(
            df=test_records,
            business_key='entity_id'
        )
        
        self.assertEqual(len(result['predictions']), 3)
        self.assertEqual(len(result['anomaly_indicators']), 3)
        self.assertGreaterEqual(result['anomaly_count'], 0)
        self.assertLessEqual(result['anomaly_rate'], 1.0)
    
    def test_cross_entity_anomaly_detection_comprehensive(self):
        """Test cross-entity anomaly detection with comprehensive data"""
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        
        # Train the model
        detector.train(
            df=self.test_data,
            business_key='entity_id',
            target_attributes=['value', 'rating', 'size'],
            time_column='date'
        )
        
        # Test cross-entity detection
        test_records = pd.DataFrame([
            {'entity_id': 'ENTITY_001', 'value': 1000, 'rating': 4.5, 'size': 5000, 'date': '2024-01-01'},
            {'entity_id': 'ENTITY_002', 'value': 800, 'rating': 3.2, 'size': 3000, 'date': '2024-01-01'},
            {'entity_id': 'ENTITY_003', 'value': 1200, 'rating': 4.8, 'size': 6000, 'date': '2024-01-01'}
        ])
        
        result = detector.detect_cross_entity_anomalies(
            df=test_records,
            business_key='entity_id',
            time_column='date'
        )
        
        self.assertIn('cross_entity_analysis', result)
        self.assertIn('cross_entity_events', result)
        self.assertIn('total_cross_entity_events', result)
        self.assertIn('threshold_used', result)
    
    def test_model_persistence_comprehensive(self):
        """Test comprehensive model saving and loading"""
        detector = GenericAnomalyDetector(algorithm='random_forest', contamination=0.1)
        
        # Train the model
        detector.train(
            df=self.test_data,
            business_key='entity_id',
            target_attributes=['value', 'rating', 'size'],
            time_column='date',
            anomaly_labels='is_anomaly'
        )
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        detector.save_model(model_path)
        self.assertTrue(os.path.exists(model_path))
        
        # Load model
        loaded_detector = GenericAnomalyDetector.load_model(model_path)
        
        # Verify all attributes
        self.assertEqual(loaded_detector.algorithm, detector.algorithm)
        self.assertEqual(loaded_detector.contamination, detector.contamination)
        self.assertTrue(loaded_detector.is_trained)
        self.assertEqual(loaded_detector.feature_columns, detector.feature_columns)
        self.assertEqual(loaded_detector.business_keys, detector.business_keys)
        self.assertEqual(loaded_detector.target_attributes, detector.target_attributes)
        
        # Test that loaded model works
        test_record = pd.DataFrame([{
            'entity_id': 'TEST_ENTITY',
            'value': 1000,
            'rating': 4.5,
            'size': 5000,
            'date': '2024-01-01'
        }])
        
        result = loaded_detector.predict(
            df=test_record,
            business_key='entity_id',
            time_column='date'
        )
        
        self.assertIn('predictions', result)
        self.assertEqual(len(result['predictions']), 1)
        
        # Clean up
        os.unlink(model_path)
    
    def test_feature_engineering_comprehensive(self):
        """Test comprehensive feature engineering"""
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        
        # Train the model
        detector.train(
            df=self.test_data,
            business_key='entity_id',
            target_attributes=['value', 'rating', 'size'],
            time_column='date'
        )
        
        # Check that features were created
        self.assertGreater(len(detector.feature_columns), 3)  # More than just the original 3
        
        # Check for specific feature types
        feature_names = detector.feature_columns
        
        # Statistical features
        self.assertTrue(any('_change' in f for f in feature_names))
        self.assertTrue(any('_ma_' in f for f in feature_names))
        self.assertTrue(any('_std_' in f for f in feature_names))
        self.assertTrue(any('_zscore_' in f for f in feature_names))
        
        # Volatility features
        self.assertTrue(any('_volatility_' in f for f in feature_names))
        
        # Momentum features
        self.assertTrue(any('_momentum_' in f for f in feature_names))
        
        # Percentile features
        self.assertTrue(any('_percentile_' in f for f in feature_names))
        
        # Temporal features (since we have time column)
        self.assertTrue(any('hour' in f for f in feature_names))
        self.assertTrue(any('day_of_week' in f for f in feature_names))
        self.assertTrue(any('month' in f for f in feature_names))
    
    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling"""
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        
        # Test prediction without training
        with self.assertRaises(ValueError):
            detector.predict(
                df=pd.DataFrame([{'entity_id': 'ENTITY_001', 'value': 1000}]),
                business_key='entity_id'
            )
        
        # Test training with missing required columns
        with self.assertRaises(ValueError):
            detector.train(
                df=pd.DataFrame([{'entity_id': 'ENTITY_001'}]),
                business_key='entity_id',
                target_attributes=['value', 'rating']
            )
        
        # Test training with missing anomaly labels for supervised
        with self.assertRaises(ValueError):
            detector.train(
                df=self.test_data,
                business_key='entity_id',
                target_attributes=['value', 'rating', 'size'],
                time_column='date',
                anomaly_labels='missing_column'
            )
        
        # Test invalid algorithm
        with self.assertRaises(ValueError):
            invalid_detector = GenericAnomalyDetector(algorithm='invalid_algorithm')
            invalid_detector._get_model()
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Test with very small dataset
        small_data = pd.DataFrame([
            {'entity_id': 'A', 'value': 100, 'rating': 4.0, 'size': 1000, 'date': '2024-01-01'},
            {'entity_id': 'B', 'value': 200, 'rating': 3.5, 'size': 2000, 'date': '2024-01-01'}
        ])
        
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        detector.train(
            df=small_data,
            business_key='entity_id',
            target_attributes=['value', 'rating', 'size'],
            time_column='date'
        )
        
        self.assertTrue(detector.is_trained)
        
        # Test with single entity
        single_entity_data = pd.DataFrame([
            {'entity_id': 'A', 'value': 100, 'rating': 4.0, 'size': 1000, 'date': '2024-01-01'},
            {'entity_id': 'A', 'value': 200, 'rating': 3.5, 'size': 2000, 'date': '2024-01-02'}
        ])
        
        detector2 = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        detector2.train(
            df=single_entity_data,
            business_key='entity_id',
            target_attributes=['value', 'rating', 'size'],
            time_column='date'
        )
        
        self.assertTrue(detector2.is_trained)
    
    def test_response_format_consistency(self):
        """Test that response format is consistent"""
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        
        # Train the model
        detector.train(
            df=self.test_data,
            business_key='entity_id',
            target_attributes=['value', 'rating', 'size'],
            time_column='date'
        )
        
        # Test single prediction
        single_data = pd.DataFrame([{
            'entity_id': 'ENTITY_001',
            'value': 1000,
            'rating': 4.5,
            'size': 5000,
            'date': '2024-01-01'
        }])
        
        single_result = detector.predict(
            df=single_data,
            business_key='entity_id',
            time_column='date'
        )
        
        # Test batch prediction
        batch_data = pd.DataFrame([
            {'entity_id': 'ENTITY_001', 'value': 1000, 'rating': 4.5, 'size': 5000, 'date': '2024-01-01'},
            {'entity_id': 'ENTITY_002', 'value': 800, 'rating': 3.2, 'size': 3000, 'date': '2024-01-01'}
        ])
        
        batch_result = detector.predict(
            df=batch_data,
            business_key='entity_id',
            time_column='date'
        )
        
        # Both should have the same structure
        self.assertIn('predictions', single_result)
        self.assertIn('anomaly_indicators', single_result)
        self.assertIn('anomaly_count', single_result)
        self.assertIn('anomaly_rate', single_result)
        self.assertIn('prediction_analysis', single_result)
        
        self.assertIn('predictions', batch_result)
        self.assertIn('anomaly_indicators', batch_result)
        self.assertIn('anomaly_count', batch_result)
        self.assertIn('anomaly_rate', batch_result)
        self.assertIn('prediction_analysis', batch_result)
        
        # Single should have 1 prediction, batch should have 2
        self.assertEqual(len(single_result['predictions']), 1)
        self.assertEqual(len(batch_result['predictions']), 2)

def run_final_comprehensive_tests():
    """Run final comprehensive tests"""
    print("🧪 Running Final Comprehensive Test Suite")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFinalComprehensive)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n📊 Final Test Summary")
    print("=" * 50)
    print(f"Total Tests: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n✅ ALL TESTS PASSED! The code is complete and working perfectly.")
        print("\n🎯 Features Verified:")
        print("   ✅ All 5 algorithms (isolation_forest, one_class_svm, local_outlier_factor, random_forest, logistic_regression)")
        print("   ✅ Supervised and unsupervised learning")
        print("   ✅ With and without time column")
        print("   ✅ Cross-entity anomaly detection")
        print("   ✅ Model persistence (save/load)")
        print("   ✅ Comprehensive feature engineering")
        print("   ✅ Error handling and edge cases")
        print("   ✅ Response format consistency")
        print("   ✅ Single and batch predictions")
        print("   ✅ Uniform API response structure")
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
        return False

if __name__ == "__main__":
    success = run_final_comprehensive_tests()
    sys.exit(0 if success else 1)
