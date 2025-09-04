#!/usr/bin/env python3
"""
Generate realistic instrument pricing data for testing anomaly detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

def generate_realistic_stock_data(symbol="AAPL", days=365, add_anomalies=True):
    """Generate realistic stock price data with anomalies."""
    np.random.seed(42)
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Remove weekends (stock markets closed)
    dates = dates[dates.weekday < 5]
    
    n_days = len(dates)
    
    # Generate realistic price movement
    initial_price = 150.0
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns ~0.05% mean, 2% std
    
    # Add some trend and volatility clustering
    trend = np.linspace(0, 0.1, n_days)  # 10% upward trend over period
    volatility = 0.01 + 0.01 * np.sin(np.arange(n_days) * 2 * np.pi / 30)  # Monthly volatility cycles
    
    # Apply trend and volatility
    returns = returns * volatility + trend / n_days
    
    # Calculate prices
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate volume (higher on volatile days)
    base_volume = 50_000_000
    volume = base_volume * (1 + np.abs(returns) * 10) * np.random.lognormal(0, 0.3, n_days)
    
    # Create OHLC data
    data = []
    for i, (date, price, vol) in enumerate(zip(dates, prices, volume)):
        # Generate realistic OHLC from close price
        daily_vol = abs(returns[i]) * price
        high = price + np.random.uniform(0, daily_vol)
        low = price - np.random.uniform(0, daily_vol)
        open_price = prices[i-1] if i > 0 else price
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(price, 2),
            'volume': int(vol),
            'symbol': symbol
        })
    
    df = pd.DataFrame(data)
    
    # Add anomalies if requested
    if add_anomalies:
        anomaly_indices = np.random.choice(len(df), size=max(1, len(df)//20), replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['spike', 'drop', 'volume_spike'])
            
            if anomaly_type == 'spike':
                df.loc[idx, 'close'] *= np.random.uniform(1.1, 1.3)
                df.loc[idx, 'high'] = max(df.loc[idx, 'high'], df.loc[idx, 'close'])
            elif anomaly_type == 'drop':
                df.loc[idx, 'close'] *= np.random.uniform(0.7, 0.9)
                df.loc[idx, 'low'] = min(df.loc[idx, 'low'], df.loc[idx, 'close'])
            elif anomaly_type == 'volume_spike':
                df.loc[idx, 'volume'] *= np.random.uniform(5, 15)
    
    return df

def generate_crypto_data(symbol="BTC", days=365):
    """Generate realistic cryptocurrency data."""
    np.random.seed(42)
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='H')  # Hourly data
    
    n_points = len(dates)
    
    # Crypto has higher volatility
    initial_price = 45000.0
    returns = np.random.normal(0.0001, 0.03, n_points)  # Higher volatility
    
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Volume varies more dramatically
    volume = np.random.lognormal(15, 1, n_points)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'price': [round(p, 2) for p in prices],
        'volume': [int(v) for v in volume],
        'symbol': symbol
    })
    
    return df

def get_yahoo_finance_data(symbol="AAPL", period="2y"):
    """Get real data from Yahoo Finance (requires yfinance)."""
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            print(f"No data found for {symbol}")
            return None
            
        # Clean up column names
        df.columns = df.columns.str.lower()
        df = df.reset_index()
        df['symbol'] = symbol
        
        return df
        
    except ImportError:
        print("yfinance not installed. Install with: pip install yfinance")
        return None

def save_sample_data():
    """Generate and save various sample datasets."""
    
    print("📊 Generating sample instrument data...")
    
    # Stock data
    stock_data = generate_realistic_stock_data("AAPL", days=500, add_anomalies=True)
    stock_data.to_csv("sample_stock_data.csv", index=False)
    print(f"✅ Generated stock data: {len(stock_data)} records")
    print(f"   Price range: ${stock_data['close'].min():.2f} - ${stock_data['close'].max():.2f}")
    print(f"   Volume range: {stock_data['volume'].min():,} - {stock_data['volume'].max():,}")
    
    # Crypto data
    crypto_data = generate_crypto_data("BTC", days=30)
    crypto_data.to_csv("sample_crypto_data.csv", index=False)
    print(f"✅ Generated crypto data: {len(crypto_data)} records")
    print(f"   Price range: ${crypto_data['price'].min():.2f} - ${crypto_data['price'].max():.2f}")
    
    # Try to get real data
    real_data = get_yahoo_finance_data("AAPL", "1y")
    if real_data is not None:
        real_data.to_csv("real_stock_data.csv", index=False)
        print(f"✅ Downloaded real AAPL data: {len(real_data)} records")
    
    print("\n📁 Sample files created:")
    print("   - sample_stock_data.csv (synthetic stock with anomalies)")
    print("   - sample_crypto_data.csv (synthetic crypto hourly data)")
    if real_data is not None:
        print("   - real_stock_data.csv (real AAPL data from Yahoo Finance)")
    
    return stock_data, crypto_data

if __name__ == "__main__":
    stock_df, crypto_df = save_sample_data()
    
    print(f"\n🔍 Sample data preview:")
    print("\nStock data (first 5 rows):")
    print(stock_df.head())
    
    print(f"\nCrypto data (first 5 rows):")
    print(crypto_df.head())
