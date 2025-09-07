#!/usr/bin/env python3
"""
Test Multi-Instrument Anomaly Detection
Single model that detects anomalies across multiple instruments
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from multi_instrument_detector import MultiInstrumentAnomalyDetector
import json

def create_multi_instrument_data():
    """Create sample data for multiple instruments."""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends
    
    # Instruments
    instruments = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    all_data = []
    
    for instrument in instruments:
        # Generate correlated price data
        base_price = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500, 'AMZN': 3000, 'TSLA': 200}[instrument]
        
        # Normal market movements
        returns = np.random.normal(0.0005, 0.02, len(dates))
        
        # Add some correlation between instruments
        if instrument != 'AAPL':
            # Other stocks correlate with AAPL
            aapl_returns = np.random.normal(0.0005, 0.02, len(dates))
            returns = 0.6 * aapl_returns + 0.4 * returns
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate volume
        volume = np.random.lognormal(15, 0.5, len(dates))
        
        # Create data for this instrument
        for i, (date, price, vol) in enumerate(zip(dates, prices, volume)):
            all_data.append({
                'date': date,
                'symbol': instrument,
                'open': price * (1 + np.random.normal(0, 0.005)),
                'high': price * (1 + abs(np.random.normal(0, 0.01))),
                'low': price * (1 - abs(np.random.normal(0, 0.01))),
                'close': price,
                'volume': int(vol)
            })
    
    return pd.DataFrame(all_data)

def create_black_swan_scenario():
    """Create a black swan event affecting all instruments."""
    df = create_multi_instrument_data()
    
    # Add a market-wide crash on a specific date
    crash_date = pd.to_datetime('2024-06-15')
    crash_indices = df[df['date'] == crash_date].index
    
    for idx in crash_indices:
        # All stocks crash by 10-20%
        crash_magnitude = np.random.uniform(0.1, 0.2)
        df.loc[idx, 'close'] *= (1 - crash_magnitude)
        df.loc[idx, 'volume'] *= np.random.uniform(2, 5)  # High volume
    
    # Add some individual anomalies
    for instrument in df['symbol'].unique():
        instrument_data = df[df['symbol'] == instrument]
        # Add 2-3 individual anomalies per instrument
        anomaly_indices = np.random.choice(instrument_data.index, size=3, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['spike', 'drop'])
            if anomaly_type == 'spike':
                df.loc[idx, 'close'] *= np.random.uniform(1.1, 1.3)
            else:
                df.loc[idx, 'close'] *= np.random.uniform(0.7, 0.9)
    
    return df

def test_multi_instrument_detection():
    """Test multi-instrument anomaly detection."""
    print("🚀 Testing Multi-Instrument Anomaly Detection")
    print("=" * 60)
    
    # Create test data
    print("📊 Creating multi-instrument test data...")
    df = create_black_swan_scenario()
    print(f"✅ Created {len(df)} records for {df['symbol'].nunique()} instruments")
    print(f"   Instruments: {sorted(df['symbol'].unique())}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Train model
    print("\n🤖 Training multi-instrument model...")
    detector = MultiInstrumentAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
    detector.train(df, target_columns=['close'])
    
    print(f"✅ Model trained successfully!")
    print(f"   Instruments: {detector.instruments}")
    print(f"   Features: {len(detector.feature_columns)}")
    print(f"   Total anomalies: {detector.training_stats['anomaly_count']}")
    print(f"   Anomaly rate: {detector.training_stats['anomaly_rate']:.2%}")
    
    # Show anomaly analysis by instrument
    print(f"\n📊 Anomaly Analysis by Instrument:")
    for instrument, stats in detector.training_stats['anomaly_analysis'].items():
        print(f"   {instrument}: {stats['anomaly_count']} anomalies ({stats['anomaly_rate']:.1%})")
    
    # Test predictions
    print(f"\n🔍 Testing predictions...")
    predictions = detector.predict(df)
    
    print(f"✅ Predictions completed!")
    print(f"   Total anomalies: {predictions['anomaly_count']}")
    print(f"   Anomaly rate: {predictions['anomaly_rate']:.2%}")
    
    # Show prediction analysis by instrument
    print(f"\n📈 Prediction Analysis by Instrument:")
    for instrument, stats in predictions['prediction_analysis'].items():
        print(f"   {instrument}: {stats['anomaly_count']} anomalies ({stats['anomaly_rate']:.1%})")
        print(f"      Avg score: {stats['avg_score']:.4f}, Max: {stats['max_score']:.4f}")
    
    # Test market-wide anomaly detection
    print(f"\n🚨 Testing Market-Wide Anomaly Detection...")
    market_analysis = detector.detect_market_wide_anomalies(df, threshold=0.3)
    
    print(f"✅ Market-wide analysis completed!")
    print(f"   Market-wide events: {market_analysis['total_market_wide_events']}")
    print(f"   Threshold used: {market_analysis['threshold_used']:.1%}")
    
    # Show market-wide events
    if market_analysis['market_wide_events']:
        print(f"\n📅 Market-Wide Events Detected:")
        for event in market_analysis['market_wide_events'][:5]:  # Show first 5
            print(f"   {event['date']}: {len(event['instruments_with_anomalies'])}/{event['total_instruments']} instruments ({event['anomaly_rate']:.1%})")
            print(f"      Affected: {', '.join(event['instruments_with_anomalies'])}")
    
    return detector, df, predictions, market_analysis

def compare_single_vs_multi_instrument():
    """Compare single instrument vs multi-instrument approach."""
    print("\n🔄 Comparing Single vs Multi-Instrument Approaches")
    print("=" * 60)
    
    # Create test data
    df = create_black_swan_scenario()
    
    # Test multi-instrument approach
    print("📊 Multi-Instrument Approach:")
    multi_detector = MultiInstrumentAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
    multi_detector.train(df, target_columns=['close'])
    multi_predictions = multi_detector.predict(df)
    
    print(f"   Instruments: {len(multi_detector.instruments)}")
    print(f"   Features: {len(multi_detector.feature_columns)}")
    print(f"   Anomalies: {multi_predictions['anomaly_count']} ({multi_predictions['anomaly_rate']:.1%})")
    
    # Test single instrument approach (using original detector)
    from anomaly_detector import AnomalyDetector
    
    print(f"\n📊 Single-Instrument Approach (per instrument):")
    single_anomalies = 0
    single_features = 0
    
    for instrument in df['symbol'].unique():
        instrument_data = df[df['symbol'] == instrument]
        
        single_detector = AnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        single_detector.train(instrument_data, target_column='close')
        single_predictions = single_detector.predict(instrument_data)
        
        single_anomalies += single_predictions['anomaly_count']
        single_features += len(single_detector.feature_columns)
        
        print(f"   {instrument}: {single_predictions['anomaly_count']} anomalies, {len(single_detector.feature_columns)} features")
    
    print(f"\n📈 Comparison Summary:")
    print(f"   Multi-Instrument:")
    print(f"     Total anomalies: {multi_predictions['anomaly_count']}")
    print(f"     Features per prediction: {len(multi_detector.feature_columns)}")
    print(f"     Models needed: 1")
    print(f"     Market-wide detection: ✅ Yes")
    
    print(f"   Single-Instrument:")
    print(f"     Total anomalies: {single_anomalies}")
    print(f"     Features per prediction: {single_features // len(df['symbol'].unique())}")
    print(f"     Models needed: {len(df['symbol'].unique())}")
    print(f"     Market-wide detection: ❌ No")

def test_cross_instrument_features():
    """Test the cross-instrument features."""
    print("\n🔍 Testing Cross-Instrument Features")
    print("=" * 60)
    
    # Create simple test data
    df = create_multi_instrument_data()
    detector = MultiInstrumentAnomalyDetector()
    
    # Create features
    features_df = detector._create_multi_instrument_features(df)
    
    print(f"✅ Features created successfully!")
    print(f"   Original columns: {len(df.columns)}")
    print(f"   Feature columns: {len(features_df.columns)}")
    
    # Show cross-instrument features
    cross_features = [col for col in features_df.columns if any(x in col for x in ['market_', 'correlation', 'breadth'])]
    print(f"\n📊 Cross-Instrument Features ({len(cross_features)}):")
    for feature in cross_features:
        print(f"   - {feature}")
    
    # Show instrument-specific features
    instrument_features = [col for col in features_df.columns if any(x in col for x in ['price_', 'volume_', 'volatility_', 'momentum_'])]
    print(f"\n📈 Instrument-Specific Features ({len(instrument_features)}):")
    for feature in instrument_features[:10]:  # Show first 10
        print(f"   - {feature}")
    if len(instrument_features) > 10:
        print(f"   ... and {len(instrument_features) - 10} more")

def main():
    """Main test workflow."""
    try:
        # Test multi-instrument detection
        detector, df, predictions, market_analysis = test_multi_instrument_detection()
        
        # Compare approaches
        compare_single_vs_multi_instrument()
        
        # Test cross-instrument features
        test_cross_instrument_features()
        
        print("\n🎉 Multi-Instrument Anomaly Detection Test Completed!")
        print("💡 Key Benefits:")
        print("   ✅ Single model for multiple instruments")
        print("   ✅ Cross-instrument feature engineering")
        print("   ✅ Market-wide anomaly detection")
        print("   ✅ Reduced model complexity and maintenance")
        print("   ✅ Better detection of systemic events")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
