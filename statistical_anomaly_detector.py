"""
Statistical Anomaly Detection Engine for Financial Time Series Data

This module implements a comprehensive statistical approach to anomaly detection
based on multiple criteria:
1. Z-score analysis (deviation from rolling mean)
2. Volatility analysis (volatility spikes and ratios)
3. Momentum analysis (cumulative returns over time)
4. Percentile analysis (extreme values in rolling windows)
5. Market-wide analysis (cross-index correlations)

The approach is designed to be interpretable and practical for financial data,
providing clear reasoning for each anomaly detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection engine for financial time series data.
    
    This detector uses multiple statistical criteria to identify anomalies:
    - Z-score analysis for individual index deviations
    - Volatility analysis for market volatility spikes
    - Momentum analysis for trend anomalies
    - Percentile analysis for extreme value detection
    - Market-wide analysis for systemic events
    """
    
    def __init__(self, 
                 z_threshold: float = 2.0,
                 momentum_threshold: float = 0.1,
                 percentile_threshold: float = 0.1,
                 volatility_threshold: float = 2.0,
                 market_threshold: float = 1.5):
        """
        Initialize the statistical anomaly detector.
        
        Args:
            z_threshold: Threshold for z-score anomalies (default: 2.0)
            momentum_threshold: Threshold for momentum anomalies (default: 0.1)
            percentile_threshold: Threshold for percentile anomalies (default: 0.1)
            volatility_threshold: Threshold for volatility anomalies (default: 2.0)
            market_threshold: Threshold for market-wide anomalies (default: 1.5)
        """
        self.z_threshold = z_threshold
        self.momentum_threshold = momentum_threshold
        self.percentile_threshold = percentile_threshold
        self.volatility_threshold = volatility_threshold
        self.market_threshold = market_threshold
        
        # Store statistics for analysis
        self.index_stats = {}
        self.market_stats = {}
        self.trained = False
        
    def calculate_statistics(self, df: pd.DataFrame, 
                           business_key: str, 
                           target_attributes: List[str], 
                           time_column: str) -> pd.DataFrame:
        """
        Calculate comprehensive statistics for anomaly detection.
        
        Args:
            df: Input DataFrame with time series data
            business_key: Column name for business entity (e.g., 'index_id')
            target_attributes: List of columns to analyze for anomalies
            time_column: Column name for time/date
            
        Returns:
            DataFrame with calculated statistics
        """
        print(f'📈 Calculating statistical features for {len(df)} records...')
        
        results = []
        
        for entity_id in df[business_key].unique():
            entity_data = df[df[business_key] == entity_id].copy()
            entity_data = entity_data.sort_values(time_column)
            
            # Calculate statistics for each target attribute
            for attr in target_attributes:
                if attr not in entity_data.columns:
                    continue
                    
                # Rolling statistics (20-day window for stability)
                window = 20
                entity_data[f'{attr}_rolling_mean'] = entity_data[attr].rolling(
                    window=window, min_periods=5).mean()
                entity_data[f'{attr}_rolling_std'] = entity_data[attr].rolling(
                    window=window, min_periods=5).std()
                entity_data[f'{attr}_z_score'] = (
                    entity_data[attr] - entity_data[f'{attr}_rolling_mean']
                ) / (entity_data[f'{attr}_rolling_std'] + 1e-8)
                
                # Volatility features
                entity_data[f'{attr}_volatility_10'] = entity_data[attr].rolling(
                    window=10, min_periods=5).std()
                entity_data[f'{attr}_volatility_30'] = entity_data[attr].rolling(
                    window=30, min_periods=5).std()
                entity_data[f'{attr}_volatility_ratio'] = (
                    entity_data[f'{attr}_volatility_10'] / 
                    (entity_data[f'{attr}_volatility_30'] + 1e-8)
                )
                
                # Momentum features
                entity_data[f'{attr}_momentum_5'] = entity_data[attr].rolling(
                    window=5, min_periods=3).sum()
                entity_data[f'{attr}_momentum_10'] = entity_data[attr].rolling(
                    window=10, min_periods=5).sum()
                
                # Percentile features
                entity_data[f'{attr}_percentile_20'] = entity_data[attr].rolling(
                    window=20, min_periods=5).rank(pct=True)
            
            results.append(entity_data)
        
        return pd.concat(results, ignore_index=True)
    
    def detect_market_wide_anomalies(self, df_with_stats: pd.DataFrame, 
                                   business_key: str, 
                                   target_attributes: List[str], 
                                   time_column: str) -> pd.DataFrame:
        """
        Detect market-wide anomalies by analyzing cross-entity correlations.
        
        Args:
            df_with_stats: DataFrame with calculated statistics
            business_key: Column name for business entity
            target_attributes: List of columns analyzed
            time_column: Column name for time/date
            
        Returns:
            DataFrame with market-wide analysis
        """
        print('🌍 Analyzing market-wide anomalies...')
        
        # Group by date to check cross-entity correlations
        daily_stats = df_with_stats.groupby(time_column).agg({
            **{f'{attr}_z_score': ['mean', 'std'] for attr in target_attributes},
            **{f'{attr}_volatility_10': 'mean' for attr in target_attributes}
        }).reset_index()
        
        # Flatten column names
        new_columns = [time_column]
        for attr in target_attributes:
            new_columns.extend([f'{attr}_market_z_mean', f'{attr}_market_z_std', f'{attr}_market_volatility'])
        
        daily_stats.columns = new_columns
        
        # Merge back to original data
        df_with_stats = df_with_stats.merge(daily_stats, on=time_column, how='left')
        
        return df_with_stats
    
    def detect_anomalies(self, df_with_stats: pd.DataFrame, 
                        business_key: str, 
                        target_attributes: List[str]) -> pd.DataFrame:
        """
        Detect anomalies using multiple statistical criteria.
        
        Args:
            df_with_stats: DataFrame with calculated statistics
            business_key: Column name for business entity
            target_attributes: List of columns analyzed
            
        Returns:
            DataFrame with anomaly classifications
        """
        print('🔍 Detecting anomalies using statistical criteria...')
        
        # Initialize anomaly flags
        all_anomalies = pd.Series([False] * len(df_with_stats), index=df_with_stats.index)
        anomaly_types = {}
        
        for attr in target_attributes:
            if attr not in df_with_stats.columns:
                continue
                
            # 1. Z-score anomalies
            z_anomaly = abs(df_with_stats[f'{attr}_z_score']) > self.z_threshold
            
            # 2. Volatility anomalies
            volatility_anomaly = (
                (df_with_stats[f'{attr}_volatility_ratio'] > self.volatility_threshold) |
                (df_with_stats[f'{attr}_volatility_10'] > df_with_stats[f'{attr}_volatility_30'] * 1.5)
            )
            
            # 3. Momentum anomalies
            momentum_anomaly = (
                (abs(df_with_stats[f'{attr}_momentum_5']) > self.momentum_threshold) |
                (abs(df_with_stats[f'{attr}_momentum_10']) > self.momentum_threshold * 1.5)
            )
            
            # 4. Percentile anomalies
            percentile_anomaly = (
                (df_with_stats[f'{attr}_percentile_20'] < self.percentile_threshold) |
                (df_with_stats[f'{attr}_percentile_20'] > (1 - self.percentile_threshold))
            )
            
            # 5. Market-wide anomalies
            market_wide_anomaly = (
                (abs(df_with_stats[f'{attr}_market_z_mean']) > self.market_threshold) |
                (df_with_stats[f'{attr}_market_z_std']) > 2.5
            )
            
            # Combine anomalies for this attribute
            attr_anomalies = z_anomaly | volatility_anomaly | momentum_anomaly | percentile_anomaly | market_wide_anomaly
            
            # Store individual anomaly types
            anomaly_types[f'{attr}_z_anomaly'] = z_anomaly
            anomaly_types[f'{attr}_volatility_anomaly'] = volatility_anomaly
            anomaly_types[f'{attr}_momentum_anomaly'] = momentum_anomaly
            anomaly_types[f'{attr}_percentile_anomaly'] = percentile_anomaly
            anomaly_types[f'{attr}_market_wide_anomaly'] = market_wide_anomaly
            
            # Combine with overall anomalies
            all_anomalies = all_anomalies | attr_anomalies
        
        # Add all anomaly flags to DataFrame
        df_with_stats['final_anomaly'] = all_anomalies
        for col_name, series in anomaly_types.items():
            df_with_stats[col_name] = series
        
        # Classify anomaly types (priority order)
        df_with_stats['anomaly_type'] = 'normal'
        
        # Priority classification
        for attr in target_attributes:
            if attr not in df_with_stats.columns:
                continue
                
            # Z-score anomalies (highest priority)
            z_mask = df_with_stats[f'{attr}_z_anomaly'] & (df_with_stats['anomaly_type'] == 'normal')
            df_with_stats.loc[z_mask, 'anomaly_type'] = f'{attr}_z_score'
            
            # Volatility anomalies
            vol_mask = df_with_stats[f'{attr}_volatility_anomaly'] & (df_with_stats['anomaly_type'] == 'normal')
            df_with_stats.loc[vol_mask, 'anomaly_type'] = f'{attr}_volatility'
            
            # Momentum anomalies
            mom_mask = df_with_stats[f'{attr}_momentum_anomaly'] & (df_with_stats['anomaly_type'] == 'normal')
            df_with_stats.loc[mom_mask, 'anomaly_type'] = f'{attr}_momentum'
            
            # Percentile anomalies
            perc_mask = df_with_stats[f'{attr}_percentile_anomaly'] & (df_with_stats['anomaly_type'] == 'normal')
            df_with_stats.loc[perc_mask, 'anomaly_type'] = f'{attr}_percentile'
            
            # Market-wide anomalies
            mkt_mask = df_with_stats[f'{attr}_market_wide_anomaly'] & (df_with_stats['anomaly_type'] == 'normal')
            df_with_stats.loc[mkt_mask, 'anomaly_type'] = f'{attr}_market_wide'
        
        return df_with_stats
    
    def train(self, df: pd.DataFrame, 
              business_key: str, 
              target_attributes: List[str], 
              time_column: str) -> Dict[str, Any]:
        """
        Train the statistical anomaly detector.
        
        Args:
            df: Training DataFrame
            business_key: Column name for business entity
            target_attributes: List of columns to analyze
            time_column: Column name for time/date
            
        Returns:
            Training results dictionary
        """
        print(f'🚀 Training Statistical Anomaly Detector...')
        print(f'   Business Key: {business_key}')
        print(f'   Target Attributes: {target_attributes}')
        print(f'   Time Column: {time_column}')
        print(f'   Records: {len(df):,}')
        
        # Calculate statistics
        df_with_stats = self.calculate_statistics(df, business_key, target_attributes, time_column)
        
        # Detect market-wide anomalies
        df_with_stats = self.detect_market_wide_anomalies(df_with_stats, business_key, target_attributes, time_column)
        
        # Detect anomalies
        df_classified = self.detect_anomalies(df_with_stats, business_key, target_attributes)
        
        # Store results
        self.trained_data = df_classified
        self.business_key = business_key
        self.target_attributes = target_attributes
        self.time_column = time_column
        self.trained = True
        
        # Analyze results
        analysis = self.analyze_results(df_classified)
        
        print(f'✅ Training complete!')
        print(f'   Anomaly Rate: {analysis["anomaly_rate"]*100:.1f}%')
        print(f'   Total Anomalies: {analysis["total_anomalies"]:,}')
        
        return {
            'anomaly_rate': analysis['anomaly_rate'],
            'total_anomalies': analysis['total_anomalies'],
            'total_records': analysis['total_records'],
            'anomaly_breakdown': analysis['anomaly_breakdown']
        }
    
    def predict(self, df: pd.DataFrame, 
                business_key: str, 
                time_column: str) -> Dict[str, Any]:
        """
        Predict anomalies for new data.
        
        Args:
            df: Input DataFrame for prediction
            business_key: Column name for business entity
            time_column: Column name for time/date
            
        Returns:
            Prediction results dictionary
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        print(f'🔍 Predicting anomalies for {len(df)} records...')
        
        # Calculate statistics for new data
        df_with_stats = self.calculate_statistics(df, business_key, self.target_attributes, time_column)
        
        # Detect market-wide anomalies
        df_with_stats = self.detect_market_wide_anomalies(df_with_stats, business_key, self.target_attributes, time_column)
        
        # Detect anomalies
        df_classified = self.detect_anomalies(df_with_stats, business_key, self.target_attributes)
        
        # Analyze results
        analysis = self.analyze_results(df_classified)
        
        # Format results for API compatibility
        predictions = df_classified['final_anomaly'].tolist()
        anomaly_indicators = ['HIGH_CONFIDENCE_ANOMALY' if x else 'NORMAL' for x in predictions]
        
        # Group by entity
        predictions_by_entity = {}
        anomaly_indicators_by_entity = {}
        confidence_breakdown_by_entity = {}
        input_attributes_by_entity = {}
        
        for entity_id in df_classified[business_key].unique():
            entity_data = df_classified[df_classified[business_key] == entity_id]
            
            predictions_by_entity[entity_id] = entity_data['final_anomaly'].tolist()
            anomaly_indicators_by_entity[entity_id] = [
                'HIGH_CONFIDENCE_ANOMALY' if x else 'NORMAL' for x in entity_data['final_anomaly']
            ]
            
            # Confidence breakdown
            total_records = len(entity_data)
            anomaly_count = entity_data['final_anomaly'].sum()
            confidence_breakdown_by_entity[entity_id] = {
                'high_confidence_anomalies': anomaly_count,
                'medium_confidence_anomalies': 0,
                'low_confidence_anomalies': 0,
                'normal_records': total_records - anomaly_count
            }
            
            # Input attributes (original data)
            input_attributes_by_entity[entity_id] = {
                attr: entity_data[attr].tolist() for attr in self.target_attributes
            }
        
        return {
            'predictions': predictions,
            'anomaly_indicators': anomaly_indicators,
            'predictions_by_entity': predictions_by_entity,
            'anomaly_indicators_by_entity': anomaly_indicators_by_entity,
            'anomaly_count': analysis['total_anomalies'],
            'anomaly_rate': analysis['anomaly_rate'],
            'prediction_analysis': {
                entity_id: {
                    'anomaly_count': confidence_breakdown_by_entity[entity_id]['high_confidence_anomalies'],
                    'anomaly_rate': confidence_breakdown_by_entity[entity_id]['high_confidence_anomalies'] / 
                                   (confidence_breakdown_by_entity[entity_id]['high_confidence_anomalies'] + 
                                    confidence_breakdown_by_entity[entity_id]['normal_records']),
                    'confidence_breakdown': confidence_breakdown_by_entity[entity_id],
                    'input_attributes': input_attributes_by_entity[entity_id]
                }
                for entity_id in predictions_by_entity.keys()
            }
        }
    
    def analyze_results(self, df_classified: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze and summarize anomaly detection results.
        
        Args:
            df_classified: DataFrame with anomaly classifications
            
        Returns:
            Analysis results dictionary
        """
        total_records = len(df_classified)
        total_anomalies = df_classified['final_anomaly'].sum()
        anomaly_rate = total_anomalies / total_records
        
        # Anomaly type breakdown
        anomaly_breakdown = df_classified['anomaly_type'].value_counts().to_dict()
        
        # Per-entity breakdown
        entity_breakdown = {}
        if hasattr(self, 'business_key'):
            for entity_id in df_classified[self.business_key].unique():
                entity_data = df_classified[df_classified[self.business_key] == entity_id]
                entity_breakdown[entity_id] = {
                    'total_records': len(entity_data),
                    'anomalies': entity_data['final_anomaly'].sum(),
                    'anomaly_rate': entity_data['final_anomaly'].sum() / len(entity_data)
                }
        
        return {
            'total_records': total_records,
            'total_anomalies': total_anomalies,
            'anomaly_rate': anomaly_rate,
            'anomaly_breakdown': anomaly_breakdown,
            'entity_breakdown': entity_breakdown
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and configuration.
        
        Returns:
            Model information dictionary
        """
        return {
            'algorithm': 'statistical',
            'z_threshold': self.z_threshold,
            'momentum_threshold': self.momentum_threshold,
            'percentile_threshold': self.percentile_threshold,
            'volatility_threshold': self.volatility_threshold,
            'market_threshold': self.market_threshold,
            'trained': self.trained,
            'target_attributes': self.target_attributes if self.trained else None,
            'business_key': self.business_key if self.trained else None,
            'time_column': self.time_column if self.trained else None
        }


def create_statistical_detector(config: Dict[str, Any]) -> StatisticalAnomalyDetector:
    """
    Create a statistical anomaly detector with custom configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured StatisticalAnomalyDetector instance
    """
    return StatisticalAnomalyDetector(
        z_threshold=config.get('z_threshold', 2.0),
        momentum_threshold=config.get('momentum_threshold', 0.1),
        percentile_threshold=config.get('percentile_threshold', 0.1),
        volatility_threshold=config.get('volatility_threshold', 2.0),
        market_threshold=config.get('market_threshold', 1.5)
    )


if __name__ == "__main__":
    # Example usage
    print("📊 Statistical Anomaly Detector")
    print("=" * 50)
    
    # Load sample data
    df = pd.read_csv('index_anomaly_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Create detector
    detector = StatisticalAnomalyDetector(
        z_threshold=2.0,
        momentum_threshold=0.1,
        percentile_threshold=0.1,
        volatility_threshold=2.0
    )
    
    # Train
    results = detector.train(
        df=df,
        business_key='index_id',
        target_attributes=['index_return', 'num_constituents'],
        time_column='date'
    )
    
    print(f"\\nTraining Results:")
    print(f"  Anomaly Rate: {results['anomaly_rate']*100:.1f}%")
    print(f"  Total Anomalies: {results['total_anomalies']:,}")
    
    # Test prediction
    sample_data = df.sample(n=100, random_state=42)
    predictions = detector.predict(sample_data, 'index_id', 'date')
    
    print(f"\\nPrediction Results:")
    print(f"  Anomaly Count: {predictions['anomaly_count']}")
    print(f"  Anomaly Rate: {predictions['anomaly_rate']*100:.1f}%")
