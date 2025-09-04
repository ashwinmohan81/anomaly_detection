#!/usr/bin/env python3
"""
Enhanced Anomaly Detection for Market-Wide Events and Black Swan Detection
"""

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
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import json
import os
import requests
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class EnhancedAnomalyDetector:
    def __init__(self, algorithm: str = 'isolation_forest', contamination: float = 0.1):
        """
        Enhanced anomaly detector for market-wide events.
        
        Args:
            algorithm: 'isolation_forest', 'one_class_svm', 'local_outlier_factor', 'random_forest', 'logistic_regression'
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
        
        # Enhanced features
        self.market_features = []
        self.correlation_features = []
        self.volatility_features = []
        self.sector_mapping = {}
        
    def _get_market_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Get market data for multiple symbols (mock implementation)."""
        # In real implementation, use yfinance, Alpha Vantage, etc.
        np.random.seed(42)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]  # Remove weekends
        
        market_data = []
        for symbol in symbols:
            # Generate correlated market data
            base_price = 100 if symbol == 'SPY' else 150
            returns = np.random.normal(0.0005, 0.02, len(dates))
            
            # Add market correlation
            if symbol != 'SPY':
                spy_correlation = 0.7  # Most stocks correlate with SPY
                spy_returns = np.random.normal(0.0005, 0.02, len(dates))
                returns = spy_correlation * spy_returns + (1 - spy_correlation) * returns
            
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            for date, price in zip(dates, prices):
                market_data.append({
                    'date': date,
                    'symbol': symbol,
                    'close': round(price, 2),
                    'volume': np.random.lognormal(15, 1)
                })
        
        return pd.DataFrame(market_data)
    
    def _create_market_features(self, df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create market-wide features."""
        features_df = df.copy()
        
        # Pivot market data
        market_pivot = market_data.pivot(index='date', columns='symbol', values='close')
        
        # Calculate market indices
        if 'SPY' in market_pivot.columns:
            features_df['spy_price'] = features_df['date'].map(market_pivot['SPY'])
            features_df['spy_returns'] = features_df['spy_price'].pct_change()
            features_df['spy_volatility'] = features_df['spy_returns'].rolling(5).std()
        
        # VIX-like volatility index (mock)
        features_df['vix'] = 15 + 5 * np.random.random(len(features_df))
        
        # Market breadth (percentage of stocks up)
        features_df['market_breadth'] = 0.5 + 0.3 * np.random.random(len(features_df))
        
        # Market correlation (simplified)
        if len(market_pivot.columns) > 1:
            # Calculate average correlation with other stocks
            correlations = []
            for _, row in features_df.iterrows():
                date = row['date']
                if date in market_pivot.index:
                    # Simple correlation calculation
                    stock_prices = market_pivot.loc[date]
                    if len(stock_prices) > 1:
                        # Use price correlation as proxy
                        avg_corr = 0.7  # Typical market correlation
                        correlations.append(avg_corr)
                    else:
                        correlations.append(0.5)
                else:
                    correlations.append(0.5)
            features_df['market_correlation'] = correlations
        
        return features_df
    
    def _create_enhanced_features(self, df: pd.DataFrame, market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create enhanced features including market context."""
        features_df = df.copy()
        
        # Detect time column
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
        
        # Sort by time
        if time_column:
            features_df = features_df.sort_values(time_column)
        
        # Add market features if market data provided
        if market_data is not None:
            features_df = self._create_market_features(features_df, market_data)
        
        # Enhanced lag features
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != self.target_column and col not in ['spy_price', 'vix', 'market_breadth', 'market_correlation']:
                # Extended lag features
                for lag in [1, 2, 3, 5, 7, 10]:
                    features_df[f'{col}_lag_{lag}'] = features_df[col].shift(lag)
                
                # Rolling statistics with more windows
                for window in [3, 5, 7, 10, 14, 21]:
                    features_df[f'{col}_ma_{window}'] = features_df[col].rolling(window, min_periods=1).mean()
                    features_df[f'{col}_std_{window}'] = features_df[col].rolling(window, min_periods=1).std()
                    features_df[f'{col}_zscore_{window}'] = (features_df[col] - features_df[f'{col}_ma_{window}']) / features_df[f'{col}_std_{window}'].replace(0, 1)
                
                # Volatility features
                features_df[f'{col}_volatility_5d'] = features_df[col].rolling(5).std()
                features_df[f'{col}_volatility_20d'] = features_df[col].rolling(20).std()
                features_df[f'{col}_vol_ratio'] = features_df[f'{col}_volatility_5d'] / features_df[f'{col}_volatility_20d']
                
                # Momentum features
                features_df[f'{col}_momentum_5d'] = features_df[col] / features_df[col].shift(5) - 1
                features_df[f'{col}_momentum_20d'] = features_df[col] / features_df[col].shift(20) - 1
        
        # Market-relative features
        if 'spy_price' in features_df.columns and self.target_column in features_df.columns:
            features_df['relative_strength'] = features_df[self.target_column] / features_df['spy_price']
            features_df['relative_returns'] = features_df['relative_strength'].pct_change()
            features_df['beta_estimate'] = features_df['relative_returns'].rolling(20).corr(features_df['spy_returns'])
        
        # Time-based features
        if time_column:
            features_df['hour'] = features_df[time_column].dt.hour
            features_df['day_of_week'] = features_df[time_column].dt.dayofweek
            features_df['day_of_month'] = features_df[time_column].dt.day
            features_df['month'] = features_df[time_column].dt.month
            features_df['quarter'] = features_df[time_column].dt.quarter
            features_df['is_month_end'] = features_df[time_column].dt.is_month_end
            features_df['is_quarter_end'] = features_df[time_column].dt.is_quarter_end
        
        # Fill NaN values
        features_df = features_df.fillna(0)
        
        return features_df
    
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market regime (bull, bear, volatile)."""
        features_df = df.copy()
        
        # Simple regime detection based on volatility and trend
        if 'spy_volatility' in features_df.columns:
            # High volatility regime
            features_df['high_vol_regime'] = (features_df['spy_volatility'] > features_df['spy_volatility'].quantile(0.8)).astype(int)
            
            # Trend regime
            if 'spy_returns' in features_df.columns:
                features_df['bull_market'] = (features_df['spy_returns'].rolling(20).mean() > 0).astype(int)
                features_df['bear_market'] = (features_df['spy_returns'].rolling(20).mean() < -0.01).astype(int)
        
        return features_df
    
    def _get_model(self):
        """Get the appropriate model based on algorithm."""
        if self.algorithm == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=200,  # More trees for better performance
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
                n_estimators=200,
                random_state=42,
                class_weight='balanced',
                max_depth=10
            )
        elif self.algorithm == 'logistic_regression':
            return LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train(self, df: pd.DataFrame, target_column: str, 
              feature_columns: Optional[List[str]] = None,
              label_column: Optional[str] = None,
              market_symbols: Optional[List[str]] = None,
              start_date: Optional[str] = None,
              end_date: Optional[str] = None):
        """
        Train the enhanced anomaly detection model.
        
        Args:
            df: Training data
            target_column: Column to detect anomalies in
            feature_columns: List of feature columns (if None, auto-select)
            label_column: Column with true anomaly labels (0=normal, 1=anomaly) for supervised learning
            market_symbols: List of market symbols for market context (e.g., ['SPY', 'QQQ', 'IWM'])
            start_date: Start date for market data
            end_date: End date for market data
        """
        logging.info(f"Training enhanced {self.algorithm} model on {len(df)} records")
        
        self.target_column = target_column
        self.label_column = label_column
        self.is_supervised = label_column is not None
        
        # Get market data if symbols provided
        market_data = None
        if market_symbols and start_date and end_date:
            logging.info(f"Fetching market data for symbols: {market_symbols}")
            market_data = self._get_market_data(market_symbols, start_date, end_date)
        
        # Create enhanced features
        features_df = self._create_enhanced_features(df, market_data)
        
        # Detect market regime
        features_df = self._detect_market_regime(features_df)
        
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
        self.scaler = RobustScaler()  # More robust to outliers
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
                'predicted_anomalies': int(np.sum(y_pred)),
                'market_features': len([col for col in self.feature_columns if any(market_feat in col for market_feat in ['spy', 'vix', 'market', 'relative'])])
            }
            
            logging.info(f"Enhanced supervised training completed. F1-score: {f1:.3f}, Market features: {self.training_stats['market_features']}")
            
        else:
            # Unsupervised learning
            self.model.fit(X_scaled)
            
            # Calculate unsupervised performance
            predictions = self.model.predict(X_scaled)
            anomaly_count = np.sum(predictions == -1)
            
            # Analyze anomaly patterns
            anomaly_analysis = self._analyze_anomaly_patterns(features_df, predictions)
            
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
                'anomaly_rate': float(anomaly_count / len(df)),
                'market_features': len([col for col in self.feature_columns if any(market_feat in col for market_feat in ['spy', 'vix', 'market', 'relative'])]),
                'anomaly_analysis': anomaly_analysis
            }
            
            logging.info(f"Enhanced unsupervised training completed. Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%), Market features: {self.training_stats['market_features']}")
        
        self.is_trained = True
    
    def _analyze_anomaly_patterns(self, features_df: pd.DataFrame, predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze patterns in detected anomalies."""
        anomaly_mask = predictions == -1
        normal_mask = predictions == 1
        
        analysis = {}
        
        # Market regime analysis
        if 'high_vol_regime' in features_df.columns:
            vol_anomalies = np.sum(anomaly_mask & (features_df['high_vol_regime'] == 1))
            analysis['anomalies_in_high_vol'] = int(vol_anomalies)
            analysis['anomaly_vol_ratio'] = float(vol_anomalies / np.sum(anomaly_mask)) if np.sum(anomaly_mask) > 0 else 0
        
        # Market correlation analysis
        if 'market_correlation' in features_df.columns:
            anomaly_corr = features_df.loc[anomaly_mask, 'market_correlation'].mean()
            normal_corr = features_df.loc[normal_mask, 'market_correlation'].mean()
            analysis['avg_anomaly_correlation'] = float(anomaly_corr)
            analysis['avg_normal_correlation'] = float(normal_corr)
        
        return analysis
    
    def predict(self, df: pd.DataFrame, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Predict anomalies with enhanced market context.
        
        Args:
            df: Data to predict on
            market_data: Market context data
            
        Returns:
            Dictionary with predictions and enhanced analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create enhanced features
        features_df = self._create_enhanced_features(df, market_data)
        features_df = self._detect_market_regime(features_df)
        
        # Prepare prediction data
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        if self.is_supervised:
            predictions = self.model.predict(X_scaled)
            is_anomaly = predictions == 1
            
            if hasattr(self.model, 'predict_proba'):
                scores = self.model.predict_proba(X_scaled)[:, 1]
            else:
                scores = predictions.astype(float)
        else:
            predictions = self.model.predict(X_scaled)
            is_anomaly = predictions == -1
            
            if hasattr(self.model, 'score_samples'):
                scores = self.model.score_samples(X_scaled)
            else:
                scores = -self.model.negative_outlier_factor_
        
        # Enhanced analysis
        enhanced_analysis = self._analyze_predictions(features_df, is_anomaly, scores)
        
        return {
            'predictions': is_anomaly.tolist(),
            'scores': scores.tolist(),
            'anomaly_count': int(np.sum(is_anomaly)),
            'anomaly_rate': float(np.mean(is_anomaly)),
            'enhanced_analysis': enhanced_analysis
        }
    
    def _analyze_predictions(self, features_df: pd.DataFrame, is_anomaly: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        """Analyze predictions for market context."""
        analysis = {}
        
        if np.sum(is_anomaly) > 0:
            anomaly_indices = np.where(is_anomaly)[0]
            
            # Market regime analysis
            if 'high_vol_regime' in features_df.columns:
                vol_anomalies = np.sum(features_df.iloc[anomaly_indices]['high_vol_regime'])
                analysis['anomalies_in_high_vol_regime'] = int(vol_anomalies)
                analysis['high_vol_anomaly_ratio'] = float(vol_anomalies / len(anomaly_indices))
            
            # Market correlation analysis
            if 'market_correlation' in features_df.columns:
                anomaly_corr = features_df.iloc[anomaly_indices]['market_correlation'].mean()
                analysis['avg_anomaly_market_correlation'] = float(anomaly_corr)
            
            # Severity analysis
            anomaly_scores = scores[anomaly_indices]
            analysis['max_anomaly_score'] = float(np.max(anomaly_scores))
            analysis['avg_anomaly_score'] = float(np.mean(anomaly_scores))
            
            # Temporal clustering
            if len(anomaly_indices) > 1:
                consecutive_anomalies = np.sum(np.diff(anomaly_indices) == 1)
                analysis['consecutive_anomalies'] = int(consecutive_anomalies)
                analysis['anomaly_clustering'] = consecutive_anomalies > len(anomaly_indices) * 0.3
        
        return analysis
    
    def detect_black_swan_events(self, df: pd.DataFrame, market_data: Optional[pd.DataFrame] = None, 
                                threshold: float = 0.3) -> Dict[str, Any]:
        """
        Detect potential black swan events based on market-wide anomalies.
        
        Args:
            df: Data to analyze
            market_data: Market context data
            threshold: Threshold for considering an event as black swan (proportion of anomalies)
            
        Returns:
            Black swan event analysis
        """
        predictions = self.predict(df, market_data)
        
        black_swan_analysis = {
            'is_black_swan': False,
            'anomaly_rate': predictions['anomaly_rate'],
            'market_wide_anomaly': False,
            'high_volatility_event': False,
            'correlation_breakdown': False,
            'event_severity': 'low'
        }
        
        # Check for black swan characteristics
        if predictions['anomaly_rate'] > threshold:
            black_swan_analysis['is_black_swan'] = True
            black_swan_analysis['market_wide_anomaly'] = True
        
        enhanced_analysis = predictions.get('enhanced_analysis', {})
        
        # High volatility regime
        if enhanced_analysis.get('high_vol_anomaly_ratio', 0) > 0.5:
            black_swan_analysis['high_volatility_event'] = True
        
        # Correlation breakdown (anomalies with low market correlation)
        if enhanced_analysis.get('avg_anomaly_market_correlation', 0.5) < 0.3:
            black_swan_analysis['correlation_breakdown'] = True
        
        # Determine severity
        if predictions['anomaly_rate'] > 0.5:
            black_swan_analysis['event_severity'] = 'extreme'
        elif predictions['anomaly_rate'] > 0.3:
            black_swan_analysis['event_severity'] = 'high'
        elif predictions['anomaly_rate'] > 0.2:
            black_swan_analysis['event_severity'] = 'medium'
        
        return black_swan_analysis
    
    def save_model(self, filepath: str):
        """Save the enhanced trained model to disk."""
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
        
        logging.info(f"Enhanced model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load an enhanced trained model from disk."""
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
        
        logging.info(f"Enhanced model loaded from {filepath}")
        return detector
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get enhanced model information and statistics."""
        return {
            'is_trained': self.is_trained,
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'training_stats': self.training_stats,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'enhanced_features': True,
            'market_context': 'spy_price' in (self.feature_columns or [])
        }
