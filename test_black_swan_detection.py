#!/usr/bin/env python3
"""
Test Black Swan Event Detection with Enhanced Anomaly Detector
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enhanced_anomaly_detector import EnhancedAnomalyDetector

def create_black_swan_scenario():
    """Create a realistic black swan event scenario (Trump tariff announcement)."""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends
    
    # Normal market conditions (first 8 months)
    normal_period = dates[:200]
    # Black swan event period (September - market crash)
    event_period = dates[200:220]
    # Recovery period
    recovery_period = dates[220:]
    
    # Generate data for multiple stocks
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    all_data = []
    
    for stock in stocks:
        stock_data = []
        base_price = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500, 'AMZN': 3000, 'TSLA': 200}[stock]
        
        # Normal period
        for i, date in enumerate(normal_period):
            # Normal market movements
            daily_return = np.random.normal(0.0005, 0.02)
            if i == 0:
                price = base_price
            else:
                price = stock_data[-1]['close'] * (1 + daily_return)
            
            stock_data.append({
                'date': date,
                'symbol': stock,
                'open': price * (1 + np.random.normal(0, 0.005)),
                'high': price * (1 + abs(np.random.normal(0, 0.01))),
                'low': price * (1 - abs(np.random.normal(0, 0.01))),
                'close': price,
                'volume': np.random.lognormal(15, 0.5)
            })
        
        # Black swan event period (all stocks crash together)
        crash_date = event_period[0]
        crash_magnitude = np.random.uniform(0.15, 0.25)  # 15-25% crash
        
        for i, date in enumerate(event_period):
            if i == 0:
                # Initial crash
                price = stock_data[-1]['close'] * (1 - crash_magnitude)
            else:
                # Continued volatility
                daily_return = np.random.normal(-0.01, 0.05)  # Negative bias
                price = stock_data[-1]['close'] * (1 + daily_return)
            
            # High volume during crash
            volume = stock_data[-1]['volume'] * np.random.uniform(3, 8)
            
            stock_data.append({
                'date': date,
                'symbol': stock,
                'open': price * (1 + np.random.normal(0, 0.01)),
                'high': price * (1 + abs(np.random.normal(0, 0.02))),
                'low': price * (1 - abs(np.random.normal(0, 0.02))),
                'close': price,
                'volume': volume
            })
        
        # Recovery period
        for i, date in enumerate(recovery_period):
            # Gradual recovery with high volatility
            daily_return = np.random.normal(0.002, 0.03)  # Slight positive bias
            price = stock_data[-1]['close'] * (1 + daily_return)
            
            stock_data.append({
                'date': date,
                'symbol': stock,
                'open': price * (1 + np.random.normal(0, 0.005)),
                'high': price * (1 + abs(np.random.normal(0, 0.01))),
                'low': price * (1 - abs(np.random.normal(0, 0.01))),
                'close': price,
                'volume': np.random.lognormal(15, 0.5)
            })
        
        all_data.extend(stock_data)
    
    return pd.DataFrame(all_data)

def test_enhanced_detection():
    """Test the enhanced anomaly detector on black swan scenario."""
    print("🚀 Testing Enhanced Anomaly Detection for Black Swan Events")
    print("=" * 70)
    
    # Create black swan scenario
    print("📊 Creating black swan scenario (Trump tariff announcement)...")
    df = create_black_swan_scenario()
    print(f"✅ Created {len(df)} records for {df['symbol'].nunique()} stocks")
    
    # Test on AAPL data
    aapl_data = df[df['symbol'] == 'AAPL'].copy()
    print(f"📈 AAPL data: {len(aapl_data)} records")
    print(f"   Price range: ${aapl_data['close'].min():.2f} - ${aapl_data['close'].max():.2f}")
    
    # Create market context data (SPY, VIX-like)
    market_data = create_market_context_data(df)
    
    # Train enhanced model
    print("\n🤖 Training Enhanced Anomaly Detector...")
    detector = EnhancedAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
    
    # Get date range for market data
    start_date = aapl_data['date'].min().strftime('%Y-%m-%d')
    end_date = aapl_data['date'].max().strftime('%Y-%m-%d')
    
    detector.train(
        df=aapl_data,
        target_column='close',
        market_symbols=['SPY', 'QQQ', 'IWM'],
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"✅ Enhanced model trained successfully!")
    print(f"   Features used: {len(detector.feature_columns)}")
    print(f"   Market features: {detector.training_stats['market_features']}")
    
    # Test predictions
    print("\n🔍 Testing predictions on black swan scenario...")
    predictions = detector.predict(aapl_data, market_data)
    
    print(f"✅ Predictions completed!")
    print(f"   Total anomalies detected: {predictions['anomaly_count']}")
    print(f"   Anomaly rate: {predictions['anomaly_rate']:.2%}")
    
    # Enhanced analysis
    enhanced_analysis = predictions.get('enhanced_analysis', {})
    print(f"\n📊 Enhanced Analysis:")
    print(f"   Anomalies in high vol regime: {enhanced_analysis.get('anomalies_in_high_vol_regime', 0)}")
    print(f"   High vol anomaly ratio: {enhanced_analysis.get('high_vol_anomaly_ratio', 0):.2%}")
    print(f"   Avg anomaly market correlation: {enhanced_analysis.get('avg_anomaly_market_correlation', 0):.3f}")
    print(f"   Max anomaly score: {enhanced_analysis.get('max_anomaly_score', 0):.4f}")
    
    # Black swan detection
    print("\n🚨 Black Swan Event Detection:")
    black_swan_analysis = detector.detect_black_swan_events(aapl_data, market_data)
    
    print(f"   Is Black Swan Event: {'YES' if black_swan_analysis['is_black_swan'] else 'NO'}")
    print(f"   Market-wide anomaly: {'YES' if black_swan_analysis['market_wide_anomaly'] else 'NO'}")
    print(f"   High volatility event: {'YES' if black_swan_analysis['high_volatility_event'] else 'NO'}")
    print(f"   Correlation breakdown: {'YES' if black_swan_analysis['correlation_breakdown'] else 'NO'}")
    print(f"   Event severity: {black_swan_analysis['event_severity'].upper()}")
    
    # Analyze specific periods
    print("\n📅 Period Analysis:")
    analyze_periods(aapl_data, predictions)
    
    return detector, aapl_data, predictions, black_swan_analysis

def create_market_context_data(df):
    """Create market context data (SPY, VIX-like)."""
    dates = df['date'].unique()
    market_data = []
    
    for date in dates:
        # SPY data (market index)
        spy_price = 400 + 50 * np.sin(np.where(dates == date)[0][0] * 0.1) + np.random.normal(0, 5)
        
        # VIX-like volatility
        vix = 15 + 10 * np.random.random()
        
        market_data.append({
            'date': date,
            'symbol': 'SPY',
            'close': spy_price,
            'volume': np.random.lognormal(16, 0.5)
        })
        
        market_data.append({
            'date': date,
            'symbol': 'VIX',
            'close': vix,
            'volume': np.random.lognormal(14, 0.5)
        })
    
    return pd.DataFrame(market_data)

def analyze_periods(df, predictions):
    """Analyze anomaly patterns by time period."""
    df['is_anomaly'] = predictions['predictions']
    df['anomaly_score'] = predictions['scores']
    
    # Define periods
    normal_period = df[df['date'] < '2024-09-01']
    event_period = df[(df['date'] >= '2024-09-01') & (df['date'] < '2024-09-21')]
    recovery_period = df[df['date'] >= '2024-09-21']
    
    periods = [
        ('Normal Period', normal_period),
        ('Black Swan Event', event_period),
        ('Recovery Period', recovery_period)
    ]
    
    for period_name, period_data in periods:
        if len(period_data) > 0:
            anomaly_rate = period_data['is_anomaly'].mean()
            avg_score = period_data['anomaly_score'].mean()
            print(f"   {period_name}: {anomaly_rate:.1%} anomalies, avg score: {avg_score:.4f}")

def compare_models():
    """Compare original vs enhanced model on black swan scenario."""
    print("\n🔄 Comparing Original vs Enhanced Models...")
    
    # Create test data
    df = create_black_swan_scenario()
    aapl_data = df[df['symbol'] == 'AAPL'].copy()
    market_data = create_market_context_data(df)
    
    # Test original model (basic features)
    from anomaly_detector import AnomalyDetector
    original_detector = AnomalyDetector(algorithm='isolation_forest', contamination=0.1)
    original_detector.train(aapl_data, target_column='close')
    original_predictions = original_detector.predict(aapl_data)
    
    # Test enhanced model
    enhanced_detector = EnhancedAnomalyDetector(algorithm='isolation_forest', contamination=0.1)
    start_date = aapl_data['date'].min().strftime('%Y-%m-%d')
    end_date = aapl_data['date'].max().strftime('%Y-%m-%d')
    enhanced_detector.train(
        aapl_data, 
        target_column='close',
        market_symbols=['SPY'],
        start_date=start_date,
        end_date=end_date
    )
    enhanced_predictions = enhanced_detector.predict(aapl_data, market_data)
    
    print(f"\n📊 Model Comparison Results:")
    print(f"   Original Model:")
    print(f"     Anomalies: {original_predictions['anomaly_count']} ({original_predictions['anomaly_rate']:.1%})")
    print(f"     Features: {len(original_detector.feature_columns)}")
    
    print(f"   Enhanced Model:")
    print(f"     Anomalies: {enhanced_predictions['anomaly_count']} ({enhanced_predictions['anomaly_rate']:.1%})")
    print(f"     Features: {len(enhanced_detector.feature_columns)}")
    print(f"     Market features: {enhanced_detector.training_stats['market_features']}")
    
    # Black swan detection
    black_swan = enhanced_detector.detect_black_swan_events(aapl_data, market_data)
    print(f"     Black swan detected: {'YES' if black_swan['is_black_swan'] else 'NO'}")

def main():
    """Main test workflow."""
    try:
        # Test enhanced detection
        detector, data, predictions, black_swan = test_enhanced_detection()
        
        # Compare models
        compare_models()
        
        print("\n🎉 Enhanced Black Swan Detection Test Completed!")
        print("💡 Key Improvements:")
        print("   ✅ Market context features (SPY, VIX, correlation)")
        print("   ✅ Enhanced feature engineering (volatility, momentum)")
        print("   ✅ Market regime detection (bull/bear/volatile)")
        print("   ✅ Black swan event classification")
        print("   ✅ Temporal clustering analysis")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
