#!/usr/bin/env python3
"""
Enhanced Generic Anomaly Detection Engine
Supports mixed supervised/unsupervised data and incremental learning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import json
import pickle

class EnhancedGenericAnomalyDetector:
    def __init__(self, algorithm: str = 'isolation_forest', contamination: float = 0.1):
        """
        Enhanced generic anomaly detector that supports mixed supervised/unsupervised data.
        
        Args:
            algorithm: 'isolation_forest', 'one_class_svm', 'local_outlier_factor', 'random_forest', 'logistic_regression'
            contamination: Expected proportion of anomalies
        """
        self.algorithm = algorithm
        self.contamination = contamination
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.business_keys = []
        self.target_attributes = []
        self.is_trained = False
        self.training_stats = {}
        self.model_metadata = {}
        self.supervised_data_ratio = 0.0  # Track how much data is supervised
        
    def _create_generic_features(self, df: pd.DataFrame, 
                                business_key: str, 
                                target_attributes: List[str],
                                time_column: Optional[str] = None) -> pd.DataFrame:
        """Create features for generic anomaly detection (same as original)."""
        features_df = df.copy()
        
        # Get unique business keys
        self.business_keys = sorted(df[business_key].unique().tolist())
        
        # Create entity-specific features for each business key
        for entity in self.business_keys:
            entity_data = features_df[features_df[business_key] == entity].copy()
            
            # Sort by time if available
            if time_column and time_column in entity_data.columns:
                entity_data = entity_data.sort_values(time_column)
            
            # Create features for each target attribute
            for attr in target_attributes:
                if attr in entity_data.columns:
                    # Basic statistical features
                    entity_data[f'{attr}_change'] = entity_data[attr].pct_change()
                    entity_data[f'{attr}_ma_5'] = entity_data[attr].rolling(5, min_periods=1).mean()
                    entity_data[f'{attr}_ma_20'] = entity_data[attr].rolling(20, min_periods=1).mean()
                    entity_data[f'{attr}_std_5'] = entity_data[attr].rolling(5, min_periods=1).std()
                    entity_data[f'{attr}_std_20'] = entity_data[attr].rolling(20, min_periods=1).std()
                    entity_data[f'{attr}_zscore_5'] = (entity_data[attr] - entity_data[f'{attr}_ma_5']) / entity_data[f'{attr}_std_5'].replace(0, 1)
                    entity_data[f'{attr}_zscore_20'] = (entity_data[attr] - entity_data[f'{attr}_ma_20']) / entity_data[f'{attr}_std_20'].replace(0, 1)
                    
                    # Volatility features
                    entity_data[f'{attr}_volatility_5'] = entity_data[f'{attr}_change'].rolling(5, min_periods=1).std()
                    entity_data[f'{attr}_volatility_20'] = entity_data[f'{attr}_change'].rolling(20, min_periods=1).std()
                    entity_data[f'{attr}_volatility_ratio'] = entity_data[f'{attr}_volatility_5'] / entity_data[f'{attr}_volatility_20'].replace(0, 1)
                    entity_data[f'{attr}_volatility_ratio'] = entity_data[f'{attr}_volatility_ratio'].replace([np.inf, -np.inf], 0)
                    
                    # Momentum features
                    entity_data[f'{attr}_momentum_5'] = entity_data[attr] / entity_data[attr].shift(5) - 1
                    entity_data[f'{attr}_momentum_20'] = entity_data[attr] / entity_data[attr].shift(20) - 1
                    entity_data[f'{attr}_momentum_5'] = entity_data[f'{attr}_momentum_5'].replace([np.inf, -np.inf], 0)
                    entity_data[f'{attr}_momentum_20'] = entity_data[f'{attr}_momentum_20'].replace([np.inf, -np.inf], 0)
                    
                    # Percentile features
                    entity_data[f'{attr}_percentile_5'] = entity_data[attr].rolling(20, min_periods=1).rank(pct=True)
                    entity_data[f'{attr}_percentile_20'] = entity_data[attr].rolling(50, min_periods=1).rank(pct=True)
            
            # Update the main dataframe
            for col in entity_data.columns:
                if col not in features_df.columns:
                    features_df[col] = np.nan
                features_df.loc[features_df[business_key] == entity, col] = entity_data[col]
        
        # Create cross-entity features
        features_df = self._create_cross_entity_features(features_df, business_key, target_attributes, time_column)
        
        # Create temporal features if time column provided
        if time_column and time_column in features_df.columns:
            features_df = self._create_temporal_features(features_df, time_column)
        
        # Fill NaN values and handle infinities
        features_df = features_df.fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        return features_df
    
    def _create_cross_entity_features(self, df: pd.DataFrame, 
                                    business_key: str, 
                                    target_attributes: List[str],
                                    time_column: Optional[str] = None) -> pd.DataFrame:
        """Create features that capture relationships between business entities (same as original)."""
        features_df = df.copy()
        
        # Group by time to create cross-entity features
        if time_column and time_column in df.columns:
            group_by_col = time_column
        else:
            # If no time column, create a dummy grouping
            group_by_col = 'dummy_group'
            features_df[group_by_col] = 0
        
        # Create cross-entity features for each time period
        for time_period in features_df[group_by_col].unique():
            period_data = features_df[features_df[group_by_col] == time_period].copy()
            
            if len(period_data) < 2:  # Need at least 2 entities
                continue
            
            # Calculate cross-entity statistics for each target attribute
            for attr in target_attributes:
                if attr in period_data.columns:
                    # Cross-entity statistics
                    cross_stats = {
                        f'cross_{attr}_mean': period_data[attr].mean(),
                        f'cross_{attr}_std': period_data[attr].std(),
                        f'cross_{attr}_median': period_data[attr].median(),
                        f'cross_{attr}_min': period_data[attr].min(),
                        f'cross_{attr}_max': period_data[attr].max(),
                        f'cross_{attr}_q25': period_data[attr].quantile(0.25),
                        f'cross_{attr}_q75': period_data[attr].quantile(0.75),
                    }
                    
                    # Add cross-entity features to each entity for this time period
                    for _, row in period_data.iterrows():
                        for stat_name, stat_value in cross_stats.items():
                            features_df.loc[(features_df[group_by_col] == time_period) & 
                                          (features_df[business_key] == row[business_key]), stat_name] = stat_value
                    
                    # Relative position features
                    for _, row in period_data.iterrows():
                        relative_pos = (row[attr] - period_data[attr].min()) / (period_data[attr].max() - period_data[attr].min()) if period_data[attr].max() != period_data[attr].min() else 0.5
                        features_df.loc[(features_df[group_by_col] == time_period) & 
                                      (features_df[business_key] == row[business_key]), f'{attr}_relative_position'] = relative_pos
                        
                        # Z-score relative to cross-entity mean
                        z_score = (row[attr] - period_data[attr].mean()) / period_data[attr].std() if period_data[attr].std() != 0 else 0
                        features_df.loc[(features_df[group_by_col] == time_period) & 
                                      (features_df[business_key] == row[business_key]), f'{attr}_cross_zscore'] = z_score
        
        # Remove dummy group column if created
        if group_by_col == 'dummy_group':
            features_df = features_df.drop(columns=[group_by_col])
        
        return features_df
    
    def _create_temporal_features(self, df: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """Create temporal features from time column (same as original)."""
        features_df = df.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(features_df[time_column]):
            features_df[time_column] = pd.to_datetime(features_df[time_column], errors='coerce')
        
        # Ensure time column is timezone-naive for consistent processing
        if features_df[time_column].dt.tz is not None:
            features_df[time_column] = features_df[time_column].dt.tz_localize(None)
        
        # Handle any NaT values that might have been created
        if features_df[time_column].isna().any():
            logging.warning(f"Found {features_df[time_column].isna().sum()} NaT values in time column, filling with median")
            median_time = features_df[time_column].median()
            features_df[time_column] = features_df[time_column].fillna(median_time)
        
        # Basic temporal features
        features_df['hour'] = features_df[time_column].dt.hour
        features_df['day_of_week'] = features_df[time_column].dt.dayofweek
        features_df['day_of_month'] = features_df[time_column].dt.day
        features_df['month'] = features_df[time_column].dt.month
        features_df['quarter'] = features_df[time_column].dt.quarter
        features_df['year'] = features_df[time_column].dt.year
        features_df['is_weekend'] = (features_df[time_column].dt.dayofweek >= 5).astype(int)
        features_df['is_month_end'] = features_df[time_column].dt.is_month_end.astype(int)
        features_df['is_quarter_end'] = features_df[time_column].dt.is_quarter_end.astype(int)
        
        return features_df
    
    def _get_model(self):
        """Get the appropriate model based on algorithm (same as original)."""
        if self.algorithm == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=200,
                max_samples='auto',
                max_features=1.0
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
    
    def train(self, df: pd.DataFrame, 
              business_key: str,
              target_attributes: List[str],
              feature_columns: Optional[List[str]] = None,
              time_column: Optional[str] = None,
              anomaly_labels: Optional[str] = None):
        """
        Train the enhanced anomaly detection model with mixed supervised/unsupervised data.
        
        Args:
            df: Input DataFrame
            business_key: Column name that identifies business entities
            target_attributes: List of columns to detect anomalies in
            feature_columns: List of feature columns (if None, auto-select)
            time_column: Optional time column for temporal features
            anomaly_labels: Column name with true anomaly labels (0/1) for supervised learning
                          Can be None, or contain NaN values for unsupervised data
        """
        logging.info(f"Training enhanced {self.algorithm} model on {len(df)} records")
        
        # Validate input data
        required_cols = [business_key] + target_attributes
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for anomaly labels column
        has_labels = False
        if anomaly_labels and anomaly_labels in df.columns:
            has_labels = True
            # Check how many rows have valid labels
            valid_labels = df[anomaly_labels].notna()
            labeled_count = valid_labels.sum()
            total_count = len(df)
            self.supervised_data_ratio = labeled_count / total_count
            
            logging.info(f"Found {labeled_count}/{total_count} ({self.supervised_data_ratio:.2%}) labeled samples")
        
        # Store configuration
        self.target_attributes = target_attributes
        self.anomaly_labels = anomaly_labels
        
        # Create features
        features_df = self._create_generic_features(df, business_key, target_attributes, time_column)
        
        # Select feature columns
        if feature_columns is None:
            # Auto-select numeric columns excluding target attributes and business key
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            exclude_cols = target_attributes + [business_key]
            if time_column:
                exclude_cols.append(time_column)
            self.feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        else:
            self.feature_columns = feature_columns
        
        # Prepare training data
        X = features_df[self.feature_columns].values
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model based on data availability
        self.model = self._get_model()
        
        if has_labels and self.algorithm in ['random_forest', 'logistic_regression']:
            # Mixed supervised/unsupervised learning
            if self.supervised_data_ratio > 0.1:  # At least 10% labeled data
                # Use only labeled data for supervised learning
                labeled_mask = df[anomaly_labels].notna()
                X_labeled = X_scaled[labeled_mask]
                y_labeled = df.loc[labeled_mask, anomaly_labels].values
                
                # Convert labels: 1->1 (anomaly), 0->0 (normal)
                y_labeled = y_labeled.astype(int)
                
                self.model.fit(X_labeled, y_labeled)
                logging.info(f"Trained mixed supervised {self.algorithm} model with {len(y_labeled)} labeled samples")
            else:
                # Too few labeled samples, fall back to unsupervised
                # For supervised algorithms, we need to create pseudo-labels or use a different approach
                if self.algorithm in ['random_forest', 'logistic_regression']:
                    # Create pseudo-labels using unsupervised detection
                    temp_detector = EnhancedGenericAnomalyDetector('isolation_forest', self.contamination)
                    temp_detector._create_generic_features = self._create_generic_features
                    temp_detector._create_cross_entity_features = self._create_cross_entity_features
                    temp_detector._create_temporal_features = self._create_temporal_features
                    temp_detector.business_keys = self.business_keys
                    temp_detector.target_attributes = self.target_attributes
                    temp_detector.feature_columns = self.feature_columns
                    temp_detector.scaler = self.scaler
                    temp_detector.model = temp_detector._get_model()
                    temp_detector.model.fit(X_scaled)
                    
                    # Get pseudo-labels
                    pseudo_predictions = temp_detector.model.predict(X_scaled)
                    pseudo_labels = (pseudo_predictions == -1).astype(int)
                    
                    # Train supervised model with pseudo-labels
                    self.model.fit(X_scaled, pseudo_labels)
                    logging.info(f"Too few labeled samples ({self.supervised_data_ratio:.2%}), using pseudo-labels for {self.algorithm}")
                else:
                    self.model.fit(X_scaled)
                    logging.info(f"Trained unsupervised {self.algorithm} model")
        else:
            # Unsupervised learning
            self.model.fit(X_scaled)
            logging.info(f"Trained unsupervised {self.algorithm} model")
        
        # Calculate performance
        if has_labels and self.algorithm in ['random_forest', 'logistic_regression'] and self.supervised_data_ratio > 0.1:
            # Supervised models predict 0/1, convert to -1/1 for consistency
            predictions = self.model.predict(X_scaled)
            predictions = np.where(predictions == 1, -1, 1)  # 1->-1 (anomaly), 0->1 (normal)
        else:
            # Unsupervised models already predict -1/1
            predictions = self.model.predict(X_scaled)
        
        anomaly_count = np.sum(predictions == -1)
        
        # Analyze anomalies by business key
        anomaly_analysis = self._analyze_anomalies_by_entity(features_df, business_key, predictions)
        
        self.training_stats = {
            'n_samples': len(df),
            'n_entities': len(self.business_keys),
            'business_keys': self.business_keys,
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'target_attributes': target_attributes,
            'business_key_column': business_key,
            'time_column': time_column,
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'training_date': datetime.now().isoformat(),
            'anomalies_detected': int(anomaly_count),
            'anomaly_count': int(anomaly_count),
            'anomaly_rate': float(anomaly_count / len(df)),
            'anomaly_analysis': anomaly_analysis,
            'supervised_data_ratio': self.supervised_data_ratio,
            'has_mixed_data': has_labels and self.supervised_data_ratio < 1.0
        }
        
        self.is_trained = True
        logging.info(f"Enhanced training completed. Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)")
    
    def update_with_new_data(self, df: pd.DataFrame, 
                            business_key: str,
                            anomaly_labels: Optional[str] = None):
        """
        Update the model with new data (incremental learning).
        
        Args:
            df: New DataFrame with additional data
            business_key: Column name that identifies business entities
            anomaly_labels: Column name with true anomaly labels (0/1) for supervised learning
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before updating with new data")
        
        logging.info(f"Updating model with {len(df)} new records")
        
        # Create features for new data
        features_df = self._create_generic_features(df, business_key, self.target_attributes, None)
        
        # Prepare new data
        X_new = features_df[self.feature_columns].values
        X_new_scaled = self.scaler.transform(X_new)
        
        # Check if we have new labeled data
        has_new_labels = False
        if anomaly_labels and anomaly_labels in df.columns:
            valid_labels = df[anomaly_labels].notna()
            if valid_labels.any():
                has_new_labels = True
                y_new = df.loc[valid_labels, anomaly_labels].values.astype(int)
                X_new_labeled = X_new_scaled[valid_labels]
        
        # Update model based on algorithm and data availability
        if hasattr(self.model, 'partial_fit') and has_new_labels:
            # Online learning algorithms
            self.model.partial_fit(X_new_labeled, y_new)
            logging.info(f"Updated model with {len(y_new)} new labeled samples")
        else:
            # For algorithms that don't support online learning, retrain with combined data
            logging.warning("Algorithm doesn't support online learning, retraining with combined data")
            # This would require storing all previous data, which is not implemented here
            # In practice, you'd want to implement a data storage mechanism
    
    def predict(self, df: pd.DataFrame, business_key: str, time_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict anomalies across business entities (same as original).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create features
        features_df = self._create_generic_features(df, business_key, self.target_attributes, time_column)
        
        # Prepare prediction data
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Handle supervised vs unsupervised model outputs
        if self.algorithm in ['random_forest', 'logistic_regression']:
            # Supervised models predict 0/1, convert to -1/1 for consistency
            predictions = np.where(predictions == 1, -1, 1)  # 1->-1 (anomaly), 0->1 (normal)
        
        is_anomaly = predictions == -1
        
        # Get anomaly scores
        if hasattr(self.model, 'score_samples'):
            scores = self.model.score_samples(X_scaled)
        elif hasattr(self.model, 'predict_proba'):
            # For supervised models, use probability of anomaly class
            proba = self.model.predict_proba(X_scaled)
            if proba.shape[1] == 2:  # Binary classification
                scores = proba[:, 1]  # Probability of anomaly class
            else:
                scores = np.zeros(len(X_scaled))
        else:
            scores = -self.model.negative_outlier_factor_
        
        # Analyze predictions by entity
        prediction_analysis = self._analyze_predictions_by_entity(features_df, business_key, is_anomaly, scores)
        
        return {
            'predictions': is_anomaly.tolist(),
            'scores': scores.tolist(),
            'anomaly_count': int(np.sum(is_anomaly)),
            'anomaly_rate': float(np.mean(is_anomaly)),
            'prediction_analysis': prediction_analysis
        }
    
    def _analyze_anomalies_by_entity(self, df: pd.DataFrame, business_key: str, predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze anomalies by business entity (same as original)."""
        df['is_anomaly'] = predictions == -1
        
        analysis = {}
        for entity in self.business_keys:
            entity_data = df[df[business_key] == entity]
            entity_anomalies = entity_data['is_anomaly'].sum()
            entity_total = len(entity_data)
            
            analysis[entity] = {
                'anomaly_count': int(entity_anomalies),
                'anomaly_rate': float(entity_anomalies / entity_total) if entity_total > 0 else 0,
                'total_records': int(entity_total)
            }
        
        return analysis
    
    def _analyze_predictions_by_entity(self, df: pd.DataFrame, business_key: str, 
                                     is_anomaly: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        """Analyze predictions by business entity (same as original)."""
        df['is_anomaly'] = is_anomaly
        df['anomaly_score'] = scores
        
        analysis = {}
        for entity in df[business_key].unique():
            entity_data = df[df[business_key] == entity]
            if len(entity_data) == 0:
                continue
                
            entity_anomalies = entity_data['is_anomaly'].sum()
            entity_scores = entity_data['anomaly_score']
            
            analysis[entity] = {
                'anomaly_count': int(entity_anomalies),
                'anomaly_rate': float(entity_anomalies / len(entity_data)),
                'avg_score': float(entity_scores.mean()),
                'max_score': float(entity_scores.max()),
                'min_score': float(entity_scores.min())
            }
        
        return analysis
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'business_keys': self.business_keys,
            'target_attributes': self.target_attributes,
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'is_trained': self.is_trained,
            'training_stats': self.training_stats,
            'model_metadata': self.model_metadata,
            'supervised_data_ratio': self.supervised_data_ratio
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Enhanced model saved to {filepath}")
    
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
        detector.business_keys = model_data['business_keys']
        detector.target_attributes = model_data['target_attributes']
        detector.is_trained = model_data['is_trained']
        detector.training_stats = model_data['training_stats']
        detector.model_metadata = model_data.get('model_metadata', {})
        detector.supervised_data_ratio = model_data.get('supervised_data_ratio', 0.0)
        
        logging.info(f"Enhanced model loaded from {filepath}")
        return detector
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        return {
            'is_trained': self.is_trained,
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'business_keys': self.business_keys,
            'n_entities': len(self.business_keys),
            'target_attributes': self.target_attributes,
            'training_stats': self.training_stats,
            'feature_columns': self.feature_columns,
            'supervised_data_ratio': self.supervised_data_ratio,
            'has_mixed_data': self.training_stats.get('has_mixed_data', False),
            'enhanced': True
        }
