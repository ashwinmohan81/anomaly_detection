"""
CORRECTED Statistical Anomaly Detection Engine

This implementation FIXES the cross-index correlation logic to properly:
1. Check if other indices moved significantly (FIXED)
2. Flag only individual anomalies, not market-wide (FIXED)

The key fix is in the detect_cross_index_correlation method which now properly
analyzes daily cross-index movements and distinguishes between:
- Individual anomalies: Only one index moves significantly
- Market-wide events: Multiple indices move together (should NOT be flagged as individual anomalies)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class CorrectedStatisticalAnomalyDetector:
    """
    CORRECTED Statistical anomaly detection engine that properly implements
    cross-index correlation analysis to distinguish between individual anomalies
    and market-wide events.
    """
    
    def __init__(self, 
                 z_threshold: float = 2.0,
                 market_correlation_threshold: float = 0.6,
                 min_indices_for_market_wide: int = 2):
        """
        Initialize the corrected statistical anomaly detector.
        
        Args:
            z_threshold: Threshold for z-score anomalies (default: 2.0)
            market_correlation_threshold: Threshold for market-wide detection (default: 0.6)
            min_indices_for_market_wide: Minimum indices needed for market-wide event (default: 2)
        """
        self.z_threshold = z_threshold
        self.market_correlation_threshold = market_correlation_threshold
        self.min_indices_for_market_wide = min_indices_for_market_wide
        
        # Store statistics for analysis
        self.trained = False
        
    def calculate_statistics(self, df: pd.DataFrame, 
                           business_key: str, 
                           target_attributes: List[str], 
                           time_column: str) -> pd.DataFrame:
        """
        Calculate rolling statistics for anomaly detection.
        
        Args:
            df: Input DataFrame with time series data
            business_key: Column name for business entity (e.g., 'index_id')
            target_attributes: List of columns to analyze for anomalies
            time_column: Column name for time/date
            
        Returns:
            DataFrame with calculated statistics
        """
        print('📈 Calculating statistical features...')
        
        results = []
        
        for entity_id in df[business_key].unique():
            entity_data = df[df[business_key] == entity_id].copy()
            entity_data = entity_data.sort_values(time_column)
            
            # Calculate rolling statistics (20-day window for stability)
            window = 20
            for attr in target_attributes:
                if attr not in entity_data.columns:
                    continue
                    
                entity_data[f'{attr}_rolling_mean'] = entity_data[attr].rolling(
                    window=window, min_periods=5).mean()
                entity_data[f'{attr}_rolling_std'] = entity_data[attr].rolling(
                    window=window, min_periods=5).std()
                entity_data[f'{attr}_z_score'] = (
                    entity_data[attr] - entity_data[f'{attr}_rolling_mean']
                ) / (entity_data[f'{attr}_rolling_std'] + 1e-8)
            
            results.append(entity_data)
        
        return pd.concat(results, ignore_index=True)
    
    def detect_cross_index_correlation(self, df_with_stats: pd.DataFrame, 
                                    business_key: str, 
                                    target_attributes: List[str], 
                                    time_column: str) -> pd.DataFrame:
        """
        CORRECTED: Detect cross-index correlations to distinguish individual vs market-wide anomalies.
        
        This is the KEY FIX that implements your requirements:
        1. Check if other indices moved significantly
        2. Flag only individual anomalies, not market-wide
        
        Args:
            df_with_stats: DataFrame with calculated statistics
            business_key: Column name for business entity
            target_attributes: List of columns analyzed
            time_column: Column name for time/date
            
        Returns:
            DataFrame with corrected cross-index analysis
        """
        print('🌍 Analyzing cross-index correlations (CORRECTED LOGIC)...')
        
        # Create flags
        df_with_stats = df_with_stats.copy()
        df_with_stats['is_market_wide'] = False
        df_with_stats['market_wide_reason'] = 'none'
        df_with_stats['individual_anomaly'] = False
        
        # Group by date to analyze cross-index movements
        for date in df_with_stats[time_column].unique():
            date_data = df_with_stats[df_with_stats[time_column] == date]
            
            if len(date_data) < self.min_indices_for_market_wide:
                continue
                
            # Check each attribute for market-wide movement
            for attr in target_attributes:
                if f'{attr}_z_score' not in date_data.columns:
                    continue
                    
                z_scores = date_data[f'{attr}_z_score'].values
                
                # Count how many indices have high z-scores (absolute value > threshold)
                high_z_indices = []
                for i, z_score in enumerate(z_scores):
                    if abs(z_score) > self.z_threshold:
                        high_z_indices.append(i)
                
                high_z_count = len(high_z_indices)
                total_indices = len(z_scores)
                
                # Market-wide if more than market_correlation_threshold of indices have high z-scores
                is_market_wide = high_z_count >= max(
                    self.min_indices_for_market_wide, 
                    int(total_indices * self.market_correlation_threshold)
                )
                
                if is_market_wide:
                    # Mark all indices for this date as market-wide
                    date_mask = df_with_stats[time_column] == date
                    df_with_stats.loc[date_mask, 'is_market_wide'] = True
                    df_with_stats.loc[date_mask, 'market_wide_reason'] = f'{attr}_market_wide'
                    
                    # CRITICAL FIX: NO individual anomalies for this date (market-wide event)
                    df_with_stats.loc[date_mask, 'individual_anomaly'] = False
                    break  # One market-wide attribute is enough
                else:
                    # Check for individual anomalies (only some indices have high z-scores)
                    date_mask = df_with_stats[time_column] == date
                    
                    # Individual anomaly: high z-score AND not market-wide
                    for i, (idx, row) in enumerate(date_data.iterrows()):
                        if abs(row[f'{attr}_z_score']) > self.z_threshold:
                            # This is an individual anomaly
                            df_with_stats.loc[idx, 'individual_anomaly'] = True
        
        return df_with_stats
    
    def detect_anomalies(self, df_with_stats: pd.DataFrame, 
                        business_key: str, 
                        target_attributes: List[str]) -> pd.DataFrame:
        """
        Detect anomalies using corrected logic.
        
        Args:
            df_with_stats: DataFrame with calculated statistics
            business_key: Column name for business entity
            target_attributes: List of columns analyzed
            
        Returns:
            DataFrame with anomaly classifications
        """
        print('🔍 Detecting anomalies with CORRECTED logic...')
        
        # Final anomaly: individual anomaly OR market-wide (but with different classification)
        final_anomaly = df_with_stats['individual_anomaly'] | df_with_stats['is_market_wide']
        
        df_with_stats['final_anomaly'] = final_anomaly
        
        # Classify anomaly types
        df_with_stats['anomaly_type'] = 'normal'
        df_with_stats.loc[df_with_stats['individual_anomaly'], 'anomaly_type'] = 'individual'
        df_with_stats.loc[df_with_stats['is_market_wide'] & ~df_with_stats['individual_anomaly'], 'anomaly_type'] = 'market_wide'
        
        return df_with_stats
    
    def train(self, df: pd.DataFrame, 
              business_key: str, 
              target_attributes: List[str], 
              time_column: str) -> Dict[str, Any]:
        """
        Train the corrected statistical anomaly detector.
        
        Args:
            df: Training DataFrame
            business_key: Column name for business entity
            target_attributes: List of columns to analyze
            time_column: Column name for time/date
            
        Returns:
            Training results dictionary
        """
        print(f'🚀 Training CORRECTED Statistical Anomaly Detector...')
        print(f'   Business Key: {business_key}')
        print(f'   Target Attributes: {target_attributes}')
        print(f'   Time Column: {time_column}')
        print(f'   Records: {len(df):,}')
        
        # Calculate statistics
        df_with_stats = self.calculate_statistics(df, business_key, target_attributes, time_column)
        
        # Detect cross-index correlations (CORRECTED)
        df_with_stats = self.detect_cross_index_correlation(df_with_stats, business_key, target_attributes, time_column)
        
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
        print(f'   Individual Anomaly Rate: {analysis["individual_anomaly_rate"]*100:.1f}%')
        print(f'   Market-wide Anomaly Rate: {analysis["market_wide_anomaly_rate"]*100:.1f}%')
        print(f'   Total Anomaly Rate: {analysis["total_anomaly_rate"]*100:.1f}%')
        
        return {
            'individual_anomaly_rate': analysis['individual_anomaly_rate'],
            'market_wide_anomaly_rate': analysis['market_wide_anomaly_rate'],
            'total_anomaly_rate': analysis['total_anomaly_rate'],
            'total_anomalies': analysis['total_anomalies'],
            'total_records': analysis['total_records']
        }
    
    def predict(self, df: pd.DataFrame, 
                business_key: str, 
                time_column: str) -> Dict[str, Any]:
        """
        Predict anomalies for new data using corrected logic.
        
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
        
        # Detect cross-index correlations (CORRECTED)
        df_with_stats = self.detect_cross_index_correlation(df_with_stats, business_key, self.target_attributes, time_column)
        
        # Detect anomalies
        df_classified = self.detect_anomalies(df_with_stats, business_key, self.target_attributes)
        
        # Analyze results
        analysis = self.analyze_results(df_classified)
        
        # Format results for API compatibility
        predictions = df_classified['final_anomaly'].tolist()
        individual_predictions = df_classified['individual_anomaly'].tolist()
        market_wide_predictions = df_classified['is_market_wide'].tolist()
        
        # Group by entity
        predictions_by_entity = {}
        individual_predictions_by_entity = {}
        market_wide_predictions_by_entity = {}
        confidence_breakdown_by_entity = {}
        input_attributes_by_entity = {}
        
        for entity_id in df_classified[business_key].unique():
            entity_data = df_classified[df_classified[business_key] == entity_id]
            
            predictions_by_entity[entity_id] = entity_data['final_anomaly'].tolist()
            individual_predictions_by_entity[entity_id] = entity_data['individual_anomaly'].tolist()
            market_wide_predictions_by_entity[entity_id] = entity_data['is_market_wide'].tolist()
            
            # Confidence breakdown
            total_records = len(entity_data)
            individual_count = entity_data['individual_anomaly'].sum()
            market_wide_count = entity_data['is_market_wide'].sum()
            normal_count = total_records - individual_count - market_wide_count
            
            confidence_breakdown_by_entity[entity_id] = {
                'high_confidence_anomalies': individual_count,
                'medium_confidence_anomalies': market_wide_count,
                'low_confidence_anomalies': 0,
                'normal_records': normal_count
            }
            
            # Input attributes (original data)
            input_attributes_by_entity[entity_id] = {
                attr: entity_data[attr].tolist() for attr in self.target_attributes
            }
        
        return {
            'predictions': predictions,
            'individual_predictions': individual_predictions,
            'market_wide_predictions': market_wide_predictions,
            'predictions_by_entity': predictions_by_entity,
            'individual_predictions_by_entity': individual_predictions_by_entity,
            'market_wide_predictions_by_entity': market_wide_predictions_by_entity,
            'anomaly_count': analysis['total_anomalies'],
            'anomaly_rate': analysis['total_anomaly_rate'],
            'individual_anomaly_count': analysis['individual_anomalies'],
            'individual_anomaly_rate': analysis['individual_anomaly_rate'],
            'market_wide_anomaly_count': analysis['market_wide_anomalies'],
            'market_wide_anomaly_rate': analysis['market_wide_anomaly_rate'],
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
        individual_anomalies = df_classified['individual_anomaly'].sum()
        market_wide_anomalies = df_classified['is_market_wide'].sum()
        total_anomalies = df_classified['final_anomaly'].sum()
        
        # Anomaly type breakdown
        anomaly_breakdown = df_classified['anomaly_type'].value_counts().to_dict()
        
        return {
            'total_records': total_records,
            'individual_anomalies': individual_anomalies,
            'market_wide_anomalies': market_wide_anomalies,
            'total_anomalies': total_anomalies,
            'individual_anomaly_rate': individual_anomalies / total_records,
            'market_wide_anomaly_rate': market_wide_anomalies / total_records,
            'total_anomaly_rate': total_anomalies / total_records,
            'anomaly_breakdown': anomaly_breakdown
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and configuration.
        
        Returns:
            Model information dictionary
        """
        return {
            'algorithm': 'corrected_statistical',
            'z_threshold': self.z_threshold,
            'market_correlation_threshold': self.market_correlation_threshold,
            'min_indices_for_market_wide': self.min_indices_for_market_wide,
            'trained': self.trained,
            'target_attributes': self.target_attributes if self.trained else None,
            'business_key': self.business_key if self.trained else None,
            'time_column': self.time_column if self.trained else None
        }


def create_corrected_detector(config: Dict[str, Any]) -> CorrectedStatisticalAnomalyDetector:
    """
    Create a corrected statistical anomaly detector with custom configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured CorrectedStatisticalAnomalyDetector instance
    """
    return CorrectedStatisticalAnomalyDetector(
        z_threshold=config.get('z_threshold', 2.0),
        market_correlation_threshold=config.get('market_correlation_threshold', 0.6),
        min_indices_for_market_wide=config.get('min_indices_for_market_wide', 2)
    )


if __name__ == "__main__":
    # Example usage
    print("📊 CORRECTED Statistical Anomaly Detector")
    print("=" * 50)
    
    # Load sample data
    df = pd.read_csv('index_anomaly_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Create detector
    detector = CorrectedStatisticalAnomalyDetector(
        z_threshold=2.0,
        market_correlation_threshold=0.6
    )
    
    # Train
    results = detector.train(
        df=df,
        business_key='index_id',
        target_attributes=['index_return', 'num_constituents'],
        time_column='date'
    )
    
    print(f"\\nTraining Results:")
    print(f"  Individual Anomaly Rate: {results['individual_anomaly_rate']*100:.1f}%")
    print(f"  Market-wide Anomaly Rate: {results['market_wide_anomaly_rate']*100:.1f}%")
    print(f"  Total Anomaly Rate: {results['total_anomaly_rate']*100:.1f}%")
    
    # Test prediction
    sample_data = df.sample(n=100, random_state=42)
    predictions = detector.predict(sample_data, 'index_id', 'date')
    
    print(f"\\nPrediction Results:")
    print(f"  Individual Anomaly Count: {predictions['individual_anomaly_count']}")
    print(f"  Market-wide Anomaly Count: {predictions['market_wide_anomaly_count']}")
    print(f"  Total Anomaly Count: {predictions['anomaly_count']}")
    print(f"  Individual Anomaly Rate: {predictions['individual_anomaly_rate']*100:.1f}%")
    print(f"  Market-wide Anomaly Rate: {predictions['market_wide_anomaly_rate']*100:.1f}%")
