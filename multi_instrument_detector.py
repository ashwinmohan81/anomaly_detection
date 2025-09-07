#!/usr/bin/env python3
"""
Multi-Instrument Anomaly Detection
Single model that can detect anomalies across multiple instruments
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import json
import pickle

class MultiInstrumentAnomalyDetector:
    def __init__(self, algorithm: str = 'isolation_forest', contamination: float = 0.1):
        """
        Multi-instrument anomaly detector.
        
        Args:
            algorithm: 'isolation_forest', 'one_class_svm', 'local_outlier_factor'
            contamination: Expected proportion of anomalies
        """
        self.algorithm = algorithm
        self.contamination = contamination
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.instruments = []
        self.is_trained = False
        self.training_stats = {}
        self.model_metadata = {}
        
    def _create_multi_instrument_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for multi-instrument anomaly detection.
        """
        features_df = df.copy()
        
        # Ensure we have required columns
        required_cols = ['date', 'symbol', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by date and symbol
        features_df = features_df.sort_values(['date', 'symbol'])
        
        # Create time-based features
        features_df['hour'] = pd.to_datetime(features_df['date']).dt.hour
        features_df['day_of_week'] = pd.to_datetime(features_df['date']).dt.dayofweek
        features_df['day_of_month'] = pd.to_datetime(features_df['date']).dt.day
        features_df['month'] = pd.to_datetime(features_df['date']).dt.month
        
        # Create instrument-specific features
        for instrument in features_df['symbol'].unique():
            instrument_data = features_df[features_df['symbol'] == instrument].copy()
            
            # Price features
            instrument_data['price_change'] = instrument_data['close'].pct_change()
            instrument_data['price_ma_5'] = instrument_data['close'].rolling(5).mean()
            instrument_data['price_ma_20'] = instrument_data['close'].rolling(20).mean()
            instrument_data['price_std_5'] = instrument_data['close'].rolling(5).std()
            instrument_data['price_std_20'] = instrument_data['close'].rolling(20).std()
            instrument_data['price_zscore_5'] = (instrument_data['close'] - instrument_data['price_ma_5']) / instrument_data['price_std_5']
            instrument_data['price_zscore_20'] = (instrument_data['close'] - instrument_data['price_ma_20']) / instrument_data['price_std_20']
            
            # Volume features
            instrument_data['volume_ma_5'] = instrument_data['volume'].rolling(5).mean()
            instrument_data['volume_ma_20'] = instrument_data['volume'].rolling(20).mean()
            instrument_data['volume_ratio'] = instrument_data['volume'] / instrument_data['volume_ma_20']
            instrument_data['volume_zscore'] = (instrument_data['volume'] - instrument_data['volume_ma_20']) / instrument_data['volume'].rolling(20).std()
            
            # Volatility features
            instrument_data['volatility_5'] = instrument_data['price_change'].rolling(5).std()
            instrument_data['volatility_20'] = instrument_data['price_change'].rolling(20).std()
            instrument_data['volatility_ratio'] = instrument_data['volatility_5'] / instrument_data['volatility_20']
            
            # Momentum features
            instrument_data['momentum_5'] = instrument_data['close'] / instrument_data['close'].shift(5) - 1
            instrument_data['momentum_20'] = instrument_data['close'] / instrument_data['close'].shift(20) - 1
            
            # Update the main dataframe
            for col in ['price_change', 'price_ma_5', 'price_ma_20', 'price_std_5', 'price_std_20',
                       'price_zscore_5', 'price_zscore_20', 'volume_ma_5', 'volume_ma_20',
                       'volume_ratio', 'volume_zscore', 'volatility_5', 'volatility_20',
                       'volatility_ratio', 'momentum_5', 'momentum_20']:
                features_df.loc[features_df['symbol'] == instrument, col] = instrument_data[col]
        
        # Create cross-instrument features
        features_df = self._create_cross_instrument_features(features_df)
        
        # Create market-wide features
        features_df = self._create_market_features(features_df)
        
        # Fill NaN values
        features_df = features_df.fillna(0)
        
        return features_df
    
    def _create_cross_instrument_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that capture relationships between instruments."""
        features_df = df.copy()
        
        # Group by date to create cross-instrument features
        daily_features = []
        
        for date in df['date'].unique():
            date_data = df[df['date'] == date].copy()
            
            if len(date_data) < 2:  # Need at least 2 instruments
                continue
                
            # Market-wide statistics
            market_stats = {
                'market_avg_price': date_data['close'].mean(),
                'market_std_price': date_data['close'].std(),
                'market_avg_volume': date_data['volume'].mean(),
                'market_std_volume': date_data['volume'].std(),
                'market_avg_volatility': date_data['price_change'].mean() if 'price_change' in date_data.columns else 0,
                'market_std_volatility': date_data['price_change'].std() if 'price_change' in date_data.columns else 0,
            }
            
            # Add market features to each instrument for this date
            for _, row in date_data.iterrows():
                for stat_name, stat_value in market_stats.items():
                    features_df.loc[(features_df['date'] == date) & (features_df['symbol'] == row['symbol']), stat_name] = stat_value
            
            # Cross-instrument correlations
            if len(date_data) >= 2:
                price_corr = date_data['close'].corr(date_data['close'].shift(1))
                volume_corr = date_data['volume'].corr(date_data['volume'].shift(1))
                
                for _, row in date_data.iterrows():
                    features_df.loc[(features_df['date'] == date) & (features_df['symbol'] == row['symbol']), 'price_correlation'] = price_corr
                    features_df.loc[(features_df['date'] == date) & (features_df['symbol'] == row['symbol']), 'volume_correlation'] = volume_corr
        
        return features_df
    
    def _create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market-wide features."""
        features_df = df.copy()
        
        # Market regime detection
        if 'market_avg_price' in features_df.columns:
            # Market trend
            features_df['market_trend_5'] = features_df['market_avg_price'].rolling(5).mean()
            features_df['market_trend_20'] = features_df['market_avg_price'].rolling(20).mean()
            features_df['market_trend_ratio'] = features_df['market_trend_5'] / features_df['market_trend_20']
            
            # Market volatility regime
            features_df['market_vol_regime'] = (features_df['market_std_volatility'] > features_df['market_std_volatility'].quantile(0.8)).astype(int)
            
            # Market breadth (how many instruments are up)
            daily_returns = features_df.groupby('date')['price_change'].apply(lambda x: (x > 0).sum() / len(x))
            features_df['market_breadth'] = features_df['date'].map(daily_returns)
        
        return features_df
    
    def _get_model(self):
        """Get the appropriate model based on algorithm."""
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
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train(self, df: pd.DataFrame, target_columns: List[str] = ['close'], 
              feature_columns: Optional[List[str]] = None):
        """
        Train the multi-instrument anomaly detection model.
        
        Args:
            df: DataFrame with columns ['date', 'symbol', 'close', 'volume', ...]
            target_columns: Columns to detect anomalies in
            feature_columns: List of feature columns (if None, auto-select)
        """
        logging.info(f"Training multi-instrument {self.algorithm} model on {len(df)} records")
        
        # Validate input data
        required_cols = ['date', 'symbol', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Get unique instruments
        self.instruments = sorted(df['symbol'].unique().tolist())
        logging.info(f"Training on {len(self.instruments)} instruments: {self.instruments}")
        
        # Create features
        features_df = self._create_multi_instrument_features(df)
        
        # Select feature columns
        if feature_columns is None:
            # Auto-select numeric columns excluding target columns
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            exclude_cols = target_columns + ['hour', 'day_of_week', 'day_of_month', 'month']
            self.feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        else:
            self.feature_columns = feature_columns
        
        # Prepare training data
        X = features_df[self.feature_columns].values
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = self._get_model()
        self.model.fit(X_scaled)
        
        # Calculate performance
        predictions = self.model.predict(X_scaled)
        anomaly_count = np.sum(predictions == -1)
        
        # Analyze anomalies by instrument
        anomaly_analysis = self._analyze_anomalies_by_instrument(features_df, predictions)
        
        self.training_stats = {
            'n_samples': len(df),
            'n_instruments': len(self.instruments),
            'instruments': self.instruments,
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'target_columns': target_columns,
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'training_date': datetime.now().isoformat(),
            'anomalies_detected': int(anomaly_count),
            'anomaly_count': int(anomaly_count),
            'anomaly_rate': float(anomaly_count / len(df)),
            'anomaly_analysis': anomaly_analysis
        }
        
        self.is_trained = True
        logging.info(f"Multi-instrument training completed. Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)")
    
    def _analyze_anomalies_by_instrument(self, df: pd.DataFrame, predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze anomalies by instrument."""
        df['is_anomaly'] = predictions == -1
        
        analysis = {}
        for instrument in self.instruments:
            instrument_data = df[df['symbol'] == instrument]
            instrument_anomalies = instrument_data['is_anomaly'].sum()
            instrument_total = len(instrument_data)
            
            analysis[instrument] = {
                'anomaly_count': int(instrument_anomalies),
                'anomaly_rate': float(instrument_anomalies / instrument_total) if instrument_total > 0 else 0,
                'total_records': int(instrument_total)
            }
        
        return analysis
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict anomalies across multiple instruments.
        
        Args:
            df: DataFrame with columns ['date', 'symbol', 'close', 'volume', ...]
            
        Returns:
            Dictionary with predictions and analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create features
        features_df = self._create_multi_instrument_features(df)
        
        # Prepare prediction data
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        is_anomaly = predictions == -1
        
        # Get anomaly scores
        if hasattr(self.model, 'score_samples'):
            scores = self.model.score_samples(X_scaled)
        else:
            scores = -self.model.negative_outlier_factor_
        
        # Analyze predictions by instrument
        prediction_analysis = self._analyze_predictions_by_instrument(features_df, is_anomaly, scores)
        
        return {
            'predictions': is_anomaly.tolist(),
            'scores': scores.tolist(),
            'anomaly_count': int(np.sum(is_anomaly)),
            'anomaly_rate': float(np.mean(is_anomaly)),
            'prediction_analysis': prediction_analysis
        }
    
    def _analyze_predictions_by_instrument(self, df: pd.DataFrame, is_anomaly: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        """Analyze predictions by instrument."""
        df['is_anomaly'] = is_anomaly
        df['anomaly_score'] = scores
        
        analysis = {}
        for instrument in self.instruments:
            instrument_data = df[df['symbol'] == instrument]
            if len(instrument_data) == 0:
                continue
                
            instrument_anomalies = instrument_data['is_anomaly'].sum()
            instrument_scores = instrument_data['anomaly_score']
            
            analysis[instrument] = {
                'anomaly_count': int(instrument_anomalies),
                'anomaly_rate': float(instrument_anomalies / len(instrument_data)),
                'avg_score': float(instrument_scores.mean()),
                'max_score': float(instrument_scores.max()),
                'min_score': float(instrument_scores.min())
            }
        
        return analysis
    
    def detect_market_wide_anomalies(self, df: pd.DataFrame, threshold: float = 0.3) -> Dict[str, Any]:
        """
        Detect market-wide anomalies (when multiple instruments show anomalies).
        
        Args:
            df: DataFrame with columns ['date', 'symbol', 'close', 'volume', ...]
            threshold: Threshold for considering an event as market-wide (proportion of instruments)
            
        Returns:
            Market-wide anomaly analysis
        """
        predictions = self.predict(df)
        
        # Group by date to analyze market-wide patterns
        df['is_anomaly'] = predictions['predictions']
        df['anomaly_score'] = predictions['scores']
        
        market_analysis = {}
        
        for date in df['date'].unique():
            date_data = df[df['date'] == date]
            instruments_with_anomalies = date_data[date_data['is_anomaly']]['symbol'].unique()
            total_instruments = date_data['symbol'].nunique()
            
            anomaly_rate = len(instruments_with_anomalies) / total_instruments if total_instruments > 0 else 0
            
            market_analysis[date] = {
                'date': date,
                'instruments_with_anomalies': instruments_with_anomalies.tolist(),
                'anomaly_rate': anomaly_rate,
                'is_market_wide': anomaly_rate >= threshold,
                'total_instruments': total_instruments
            }
        
        # Overall market analysis
        market_wide_events = [d for d in market_analysis.values() if d['is_market_wide']]
        
        return {
            'market_analysis': market_analysis,
            'market_wide_events': market_wide_events,
            'total_market_wide_events': len(market_wide_events),
            'threshold_used': threshold
        }
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'instruments': self.instruments,
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'is_trained': self.is_trained,
            'training_stats': self.training_stats,
            'model_metadata': self.model_metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Multi-instrument model saved to {filepath}")
    
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
        detector.instruments = model_data['instruments']
        detector.is_trained = model_data['is_trained']
        detector.training_stats = model_data['training_stats']
        detector.model_metadata = model_data.get('model_metadata', {})
        
        logging.info(f"Multi-instrument model loaded from {filepath}")
        return detector
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        return {
            'is_trained': self.is_trained,
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'instruments': self.instruments,
            'n_instruments': len(self.instruments),
            'training_stats': self.training_stats,
            'feature_columns': self.feature_columns,
            'multi_instrument': True
        }
