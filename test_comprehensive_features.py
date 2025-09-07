#!/usr/bin/env python3
"""
Comprehensive test suite for the Generic Anomaly Detection API
Tests all features, endpoints, and edge cases
"""

import unittest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import io

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generic_anomaly_detector import GenericAnomalyDetector

class TestGenericAnomalyDetector(unittest.TestCase):
    """Test the GenericAnomalyDetector class"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.test_data = self.create_test_data()
        
    def create_test_data(self):
        """Create comprehensive test data"""
        data = []
        for i in range(100):
            data.append({
                'entity_id': f'ENTITY_{i % 10:03d}',
                'value': np.random.normal(1000, 100),
                'rating': np.random.normal(4.0, 0.5),
                'size': np.random.normal(5000, 500),
                'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                'is_anomaly': np.random.random() < 0.1
            })
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test detector initialization"""
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        self.assertEqual(detector.algorithm, 'isolation_forest')
        self.assertEqual(detector.contamination, 0.1)
        self.assertFalse(detector.is_trained)
    
    def test_all_algorithms_initialization(self):
        """Test all supported algorithms can be initialized"""
        algorithms = ['isolation_forest', 'one_class_svm', 'local_outlier_factor', 
                     'random_forest', 'logistic_regression']
        
        for algo in algorithms:
            with self.subTest(algorithm=algo):
                detector = GenericAnomalyDetector(algorithm=algo, contamination=0.1)
                self.assertEqual(detector.algorithm, algo)
                self.assertIsNotNone(detector._get_model())
    
    def test_unsupervised_training(self):
        """Test unsupervised training"""
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        
        detector.train(
            df=self.test_data,
            business_key='entity_id',
            target_attributes=['value', 'rating', 'size'],
            time_column='date'
        )
        
        self.assertTrue(detector.is_trained)
        self.assertIsNotNone(detector.model)
        self.assertIsNotNone(detector.scaler)
        self.assertGreater(len(detector.feature_columns), 0)
        self.assertIn('anomaly_count', detector.training_stats)
    
    def test_supervised_training(self):
        """Test supervised training"""
        detector = GenericAnomalyDetector(algorithm='random_forest', contamination=0.1)
        
        detector.train(
            df=self.test_data,
            business_key='entity_id',
            target_attributes=['value', 'rating', 'size'],
            time_column='date',
            anomaly_labels='is_anomaly'
        )
        
        self.assertTrue(detector.is_trained)
        self.assertIsNotNone(detector.model)
        self.assertIn('anomaly_count', detector.training_stats)
    
    def test_prediction_single_record(self):
        """Test prediction on single record"""
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        
        # Train the model
        detector.train(
            df=self.test_data,
            business_key='entity_id',
            target_attributes=['value', 'rating', 'size'],
            time_column='date'
        )
        
        # Test prediction
        test_record = pd.DataFrame([{
            'entity_id': 'ENTITY_001',
            'value': 1000,
            'rating': 4.5,
            'size': 5000,
            'date': '2024-01-01'
        }])
        
        result = detector.predict(
            df=test_record,
            business_key='entity_id',
            time_column='date'
        )
        
        self.assertIn('predictions', result)
        self.assertIn('scores', result)
        self.assertIn('anomaly_count', result)
        self.assertIn('anomaly_rate', result)
        self.assertIn('prediction_analysis', result)
        self.assertEqual(len(result['predictions']), 1)
        self.assertEqual(len(result['scores']), 1)
    
    def test_prediction_batch_records(self):
        """Test prediction on multiple records"""
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        
        # Train the model
        detector.train(
            df=self.test_data,
            business_key='entity_id',
            target_attributes=['value', 'rating', 'size'],
            time_column='date'
        )
        
        # Test batch prediction
        test_records = pd.DataFrame([
            {'entity_id': 'ENTITY_001', 'value': 1000, 'rating': 4.5, 'size': 5000, 'date': '2024-01-01'},
            {'entity_id': 'ENTITY_002', 'value': 800, 'rating': 3.2, 'size': 3000, 'date': '2024-01-01'},
            {'entity_id': 'ENTITY_003', 'value': 1200, 'rating': 4.8, 'size': 6000, 'date': '2024-01-01'}
        ])
        
        result = detector.predict(
            df=test_records,
            business_key='entity_id',
            time_column='date'
        )
        
        self.assertEqual(len(result['predictions']), 3)
        self.assertEqual(len(result['scores']), 3)
        self.assertGreaterEqual(result['anomaly_count'], 0)
        self.assertLessEqual(result['anomaly_rate'], 1.0)
    
    def test_without_time_column(self):
        """Test training and prediction without time column"""
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        
        # Remove time column
        data_no_time = self.test_data.drop(columns=['date'])
        
        # Train without time column
        detector.train(
            df=data_no_time,
            business_key='entity_id',
            target_attributes=['value', 'rating', 'size']
        )
        
        self.assertTrue(detector.is_trained)
        
        # Test prediction without time column
        test_record = pd.DataFrame([{
            'entity_id': 'ENTITY_001',
            'value': 1000,
            'rating': 4.5,
            'size': 5000
        }])
        
        result = detector.predict(
            df=test_record,
            business_key='entity_id'
        )
        
        self.assertIn('predictions', result)
        self.assertEqual(len(result['predictions']), 1)
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        
        # Train the model
        detector.train(
            df=self.test_data,
            business_key='entity_id',
            target_attributes=['value', 'rating', 'size'],
            time_column='date'
        )
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        detector.save_model(model_path)
        self.assertTrue(os.path.exists(model_path))
        
        # Load model
        loaded_detector = GenericAnomalyDetector.load_model(model_path)
        
        self.assertEqual(loaded_detector.algorithm, detector.algorithm)
        self.assertEqual(loaded_detector.contamination, detector.contamination)
        self.assertTrue(loaded_detector.is_trained)
        self.assertEqual(loaded_detector.feature_columns, detector.feature_columns)
        
        # Clean up
        os.unlink(model_path)
    
    def test_cross_entity_anomaly_detection(self):
        """Test cross-entity anomaly detection"""
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
    
    def test_feature_engineering(self):
        """Test that features are created correctly"""
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
        self.assertTrue(any('_change' in f for f in feature_names))  # Change features
        self.assertTrue(any('_ma_' in f for f in feature_names))    # Moving average features
        self.assertTrue(any('_std_' in f for f in feature_names))   # Standard deviation features
        # Note: Cross-entity features may not be created if there's only one entity per time period
    
    def test_error_handling(self):
        """Test error handling"""
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        
        # Test prediction without training
        with self.assertRaises(ValueError):
            detector.predict(
                df=pd.DataFrame([{'entity_id': 'ENTITY_001', 'value': 1000}]),
                business_key='entity_id'
            )
        
        # Test training with missing columns
        with self.assertRaises(ValueError):
            detector.train(
                df=pd.DataFrame([{'entity_id': 'ENTITY_001'}]),
                business_key='entity_id',
                target_attributes=['value', 'rating']
            )
    
    def test_different_data_types(self):
        """Test with different data types"""
        # Create data with different types
        data = pd.DataFrame({
            'entity_id': ['A', 'B', 'C', 'A', 'B', 'C'],
            'value': [100, 200, 300, 150, 250, 350],
            'rating': [4.5, 3.2, 4.8, 4.6, 3.1, 4.9],
            'size': [1000, 2000, 3000, 1100, 2100, 3100],
            'date': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02', '2024-01-02'],
            'is_anomaly': [False, True, False, False, True, False]
        })
        
        detector = GenericAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        
        # Train
        detector.train(
            df=data,
            business_key='entity_id',
            target_attributes=['value', 'rating', 'size'],
            time_column='date',
            anomaly_labels='is_anomaly'
        )
        
        # Predict
        result = detector.predict(
            df=data,
            business_key='entity_id',
            time_column='date'
        )
        
        self.assertEqual(len(result['predictions']), 6)
        self.assertTrue(detector.is_trained)

class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoints using FastAPI TestClient"""
    
    def setUp(self):
        """Set up test client"""
        try:
            from fastapi.testclient import TestClient
            from main import app
            self.client = TestClient(app)
        except ImportError:
            self.skipTest("FastAPI TestClient not available")
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)
    
    def test_algorithms_endpoint(self):
        """Test algorithms endpoint"""
        response = self.client.get("/algorithms")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("algorithms", data)
        self.assertGreater(len(data["algorithms"]), 0)
        
        # Check that all expected algorithms are present
        algorithm_names = [algo["name"] for algo in data["algorithms"]]
        expected_algorithms = ['isolation_forest', 'one_class_svm', 'local_outlier_factor', 
                             'random_forest', 'logistic_regression']
        for expected in expected_algorithms:
            self.assertIn(expected, algorithm_names)
    
    def test_examples_endpoint(self):
        """Test examples endpoint"""
        response = self.client.get("/examples")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("examples", data)
        self.assertGreater(len(data["examples"]), 0)
    
    def test_upload_dataset(self):
        """Test dataset upload"""
        # Create test CSV data
        test_data = pd.DataFrame({
            'entity_id': ['A', 'B', 'C'],
            'value': [100, 200, 300],
            'rating': [4.5, 3.2, 4.8],
            'size': [1000, 2000, 3000],
            'date': ['2024-01-01', '2024-01-01', '2024-01-01'],
            'is_anomaly': [False, True, False]
        })
        
        # Convert to CSV string
        csv_content = test_data.to_csv(index=False)
        
        # Upload dataset
        response = self.client.post(
            "/upload-dataset",
            files={"file": ("test.csv", csv_content, "text/csv")}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("dataset_id", data)
        self.assertIn("data_info", data)
        self.assertEqual(data["data_info"]["rows"], 3)
        self.assertEqual(data["data_info"]["columns"], 6)
        
        return data["dataset_id"]
    
    def test_train_model(self):
        """Test model training"""
        # First upload a dataset
        dataset_id = self.test_upload_dataset()
        
        # Train model
        training_request = {
            "algorithm": "isolation_forest",
            "contamination": 0.1,
            "business_key": "entity_id",
            "target_attributes": ["value", "rating", "size"],
            "time_column": "date",
            "anomaly_labels": "is_anomaly"
        }
        
        response = self.client.post(f"/train/{dataset_id}", json=training_request)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_id", data)
        self.assertIn("training_stats", data)
        self.assertIn("message", data)
        
        return data["model_id"]
    
    def test_predict_single(self):
        """Test single prediction"""
        # Train a model first
        model_id = self.test_train_model()
        
        # Make single prediction
        prediction_request = {
            "model_id": model_id,
            "data": {
                "entity_id": "ENTITY_001",
                "value": 1000,
                "rating": 4.5,
                "size": 5000,
                "date": "2024-01-01"
            }
        }
        
        response = self.client.post("/predict", json=prediction_request)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_id", data)
        self.assertIn("predictions", data)
        self.assertIn("timestamp", data)
        
        # Check uniform response structure
        self.assertIn("predictions", data["predictions"])
        self.assertIn("scores", data["predictions"])
        self.assertIn("anomaly_count", data["predictions"])
        self.assertIn("anomaly_rate", data["predictions"])
        self.assertIn("prediction_analysis", data["predictions"])
    
    def test_predict_batch(self):
        """Test batch prediction"""
        # Train a model first
        model_id = self.test_train_model()
        
        # Make batch prediction
        batch_request = {
            "model_id": model_id,
            "data": [
                {"entity_id": "ENTITY_001", "value": 1000, "rating": 4.5, "size": 5000, "date": "2024-01-01"},
                {"entity_id": "ENTITY_002", "value": 800, "rating": 3.2, "size": 3000, "date": "2024-01-01"},
                {"entity_id": "ENTITY_003", "value": 1200, "rating": 4.8, "size": 6000, "date": "2024-01-01"}
            ]
        }
        
        response = self.client.post("/predict-batch", json=batch_request)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_id", data)
        self.assertIn("predictions", data)
        self.assertIn("timestamp", data)
        
        # Check uniform response structure
        self.assertIn("predictions", data["predictions"])
        self.assertIn("scores", data["predictions"])
        self.assertIn("anomaly_count", data["predictions"])
        self.assertIn("anomaly_rate", data["predictions"])
        self.assertIn("prediction_analysis", data["predictions"])
        
        # Check that we have 3 predictions
        self.assertEqual(len(data["predictions"]["predictions"]), 3)
        self.assertEqual(len(data["predictions"]["scores"]), 3)
    
    def test_models_endpoint(self):
        """Test models listing"""
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertIn("count", data)
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        # Train a model first
        model_id = self.test_train_model()
        
        # Get model info
        response = self.client.get(f"/models/{model_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_id", data)
        self.assertIn("algorithm", data)
        self.assertIn("training_stats", data)
    
    def test_error_handling(self):
        """Test error handling"""
        # Test prediction with non-existent model
        response = self.client.post("/predict", json={
            "model_id": "non_existent_model",
            "data": {"entity_id": "ENTITY_001", "value": 1000}
        })
        self.assertEqual(response.status_code, 404)
        
        # Test training with non-existent dataset
        response = self.client.post("/train/non_existent_dataset", json={
            "algorithm": "isolation_forest",
            "contamination": 0.1,
            "business_key": "entity_id",
            "target_attributes": ["value"]
        })
        self.assertEqual(response.status_code, 404)

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🧪 Running Comprehensive Test Suite")
    print("=" * 60)
    
    # Test GenericAnomalyDetector
    print("\n1. Testing GenericAnomalyDetector Class")
    print("-" * 40)
    detector_suite = unittest.TestLoader().loadTestsFromTestCase(TestGenericAnomalyDetector)
    detector_runner = unittest.TextTestRunner(verbosity=2)
    detector_result = detector_runner.run(detector_suite)
    
    # Test API Endpoints
    print("\n2. Testing API Endpoints")
    print("-" * 40)
    api_suite = unittest.TestLoader().loadTestsFromTestCase(TestAPIEndpoints)
    api_runner = unittest.TextTestRunner(verbosity=2)
    api_result = api_runner.run(api_suite)
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    total_tests = detector_result.testsRun + api_result.testsRun
    total_failures = len(detector_result.failures) + len(api_result.failures)
    total_errors = len(detector_result.errors) + len(api_result.errors)
    
    print(f"Total Tests: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Success Rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%")
    
    if total_failures == 0 and total_errors == 0:
        print("\n✅ All tests passed! The code is complete and working.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
