import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import logging
import json
import os

class AnomalyDetector:
    def __init__(self, algorithm: str = 'isolation_forest', contamination: float = 0.1):
        """
        Initialize the anomaly detector.
        
        Args:
            algorithm: 'isolation_forest', 'one_class_svm', 'local_outlier_factor', 'random_forest', or 'logistic_regression'
            contamination: Expected proportion of anomalies
        """
        self.algorithm = algorithm
        self.contamination = contamination
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.target_column = None
        self.is_trained = False
        self.training_stats = {}
        self.model_metadata = {}
        self.is_supervised = False
        self.label_column = None
        
    def _create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Create features for anomaly detection.
        """
        features_df = df.copy()
        
        # Detect if data has time component
        date_columns = features_df.select_dtypes(include=['datetime64', 'object']).columns
        time_column = None
        
        for col in date_columns:
            if features_df[col].dtype == 'object':
                try:
                    pd.to_datetime(features_df[col].iloc[0])
                    time_column = col
                    features_df[col] = pd.to_datetime(features_df[col])
                    break
                except:
                    continue
            elif 'datetime' in str(features_df[col].dtype):
                time_column = col
                break
        
        # Sort by time if available
        if time_column:
            features_df = features_df.sort_values(time_column)
        
        # Create lag features for numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != self.target_column:
                # Lag features
                for lag in [1, 2, 3, 7]:
                    features_df[f'{col}_lag_{lag}'] = features_df[col].shift(lag)
                
                # Rolling statistics
                for window in [3, 7, 14]:
                    features_df[f'{col}_ma_{window}'] = features_df[col].rolling(window, min_periods=1).mean()
                    features_df[f'{col}_std_{window}'] = features_df[col].rolling(window, min_periods=1).std()
                    features_df[f'{col}_zscore_{window}'] = (features_df[col] - features_df[f'{col}_ma_{window}']) / features_df[f'{col}_std_{window}'].replace(0, 1)
        
        # Time-based features
        if time_column:
            features_df['hour'] = features_df[time_column].dt.hour
            features_df['day_of_week'] = features_df[time_column].dt.dayofweek
            features_df['day_of_month'] = features_df[time_column].dt.day
            features_df['month'] = features_df[time_column].dt.month
        
        # Fill NaN values
        features_df = features_df.fillna(0)
        
        return features_df
    
    def _get_model(self):
        """Get the appropriate model based on algorithm."""
        if self.algorithm == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.algorithm == 'one_class_svm':
            return OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )
        elif self.algorithm == 'local_outlier_factor':
            return LocalOutlierFactor(
                contamination=self.contamination,
                n_neighbors=20
            )
        elif self.algorithm == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        elif self.algorithm == 'logistic_regression':
            return LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train(self, df: pd.DataFrame, target_column: str, feature_columns: Optional[List[str]] = None, 
              label_column: Optional[str] = None):
        """
        Train the anomaly detection model.
        
        Args:
            df: Training data
            target_column: Column to detect anomalies in
            feature_columns: List of feature columns (if None, auto-select)
            label_column: Column with true anomaly labels (0=normal, 1=anomaly) for supervised learning
        """
        logging.info(f"Training {self.algorithm} model on {len(df)} records")
        
        self.target_column = target_column
        self.label_column = label_column
        self.is_supervised = label_column is not None
        
        # Create features
        features_df = self._create_features(df, is_training=True)
        
        # Select feature columns
        if feature_columns is None:
            # Auto-select numeric columns excluding target and label
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            exclude_cols = [target_column]
            if label_column:
                exclude_cols.append(label_column)
            self.feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        else:
            self.feature_columns = feature_columns
        
        # Prepare training data
        X = features_df[self.feature_columns].values
        
        # Scale features
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = self._get_model()
        
        if self.is_supervised:
            # Supervised learning with labels
            y = features_df[label_column].values
            self.model.fit(X_scaled, y)
            
            # Calculate supervised performance metrics
            y_pred = self.model.predict(X_scaled)
            precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
            
            self.training_stats = {
                'n_samples': len(df),
                'n_features': len(self.feature_columns),
                'feature_columns': self.feature_columns,
                'target_column': target_column,
                'label_column': label_column,
                'algorithm': self.algorithm,
                'contamination': self.contamination,
                'training_date': datetime.now().isoformat(),
                'supervised': True,
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'true_anomalies': int(np.sum(y)),
                'predicted_anomalies': int(np.sum(y_pred))
            }
            
            logging.info(f"Supervised training completed. F1-score: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
            
        else:
            # Unsupervised learning
            self.model.fit(X_scaled)
            
            # Calculate unsupervised performance
            predictions = self.model.predict(X_scaled)
            anomaly_count = np.sum(predictions == -1)
            
            self.training_stats = {
                'n_samples': len(df),
                'n_features': len(self.feature_columns),
                'feature_columns': self.feature_columns,
                'target_column': target_column,
                'algorithm': self.algorithm,
                'contamination': self.contamination,
                'training_date': datetime.now().isoformat(),
                'supervised': False,
                'anomalies_detected': int(anomaly_count),
                'anomaly_rate': float(anomaly_count / len(df))
            }
            
            logging.info(f"Unsupervised training completed. Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)")
        
        self.is_trained = True
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict anomalies in the given data.
        
        Args:
            df: Data to predict on
            
        Returns:
            Dictionary with predictions and scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create features
        features_df = self._create_features(df, is_training=False)
        
        # Prepare prediction data
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        if self.is_supervised:
            # Supervised models return 0/1
            predictions = self.model.predict(X_scaled)
            is_anomaly = predictions == 1
            
            # Get prediction probabilities for supervised models
            if hasattr(self.model, 'predict_proba'):
                scores = self.model.predict_proba(X_scaled)[:, 1]  # Probability of anomaly
            else:
                scores = predictions.astype(float)
        else:
            # Unsupervised models return -1/1
            predictions = self.model.predict(X_scaled)
            is_anomaly = predictions == -1
            
            # Get anomaly scores
            if hasattr(self.model, 'score_samples'):
                scores = self.model.score_samples(X_scaled)
            else:
                # For LOF, use negative_outlier_factor_
                scores = -self.model.negative_outlier_factor_
        
        return {
            'predictions': is_anomaly.tolist(),
            'scores': scores.tolist(),
            'anomaly_count': int(np.sum(is_anomaly)),
            'anomaly_rate': float(np.mean(is_anomaly))
        }
    
    def predict_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict anomaly for a single data point.
        
        Args:
            data: Dictionary with feature values
            
        Returns:
            Prediction result
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create single-row DataFrame
        df = pd.DataFrame([data])
        
        # Create features
        features_df = self._create_features(df, is_training=False)
        
        # Prepare prediction data
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        if self.is_supervised:
            # Supervised models return 0/1
            prediction = self.model.predict(X_scaled)[0]
            is_anomaly = prediction == 1
            
            # Get prediction probability
            if hasattr(self.model, 'predict_proba'):
                score = self.model.predict_proba(X_scaled)[0, 1]  # Probability of anomaly
            else:
                score = float(prediction)
        else:
            # Unsupervised models return -1/1
            prediction = self.model.predict(X_scaled)[0]
            is_anomaly = prediction == -1
            
            # Get score
            if hasattr(self.model, 'score_samples'):
                score = self.model.score_samples(X_scaled)[0]
            else:
                score = -self.model.negative_outlier_factor_[0]
        
        return {
            'is_anomaly': bool(is_anomaly),
            'score': float(score),
            'confidence': 'high' if abs(score) > 0.1 else 'medium' if abs(score) > 0.05 else 'low'
        }
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'is_trained': self.is_trained,
            'training_stats': self.training_stats,
            'model_metadata': self.model_metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        detector = cls(
            algorithm=model_data['algorithm'],
            contamination=model_data['contamination']
        )
        
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.feature_columns = model_data['feature_columns']
        detector.target_column = model_data['target_column']
        detector.is_trained = model_data['is_trained']
        detector.training_stats = model_data['training_stats']
        detector.model_metadata = model_data.get('model_metadata', {})
        
        logging.info(f"Model loaded from {filepath}")
        return detector
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        return {
            'is_trained': self.is_trained,
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'training_stats': self.training_stats,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
