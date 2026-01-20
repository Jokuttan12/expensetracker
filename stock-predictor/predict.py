import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

print("🚀 Loading model...")
model = load_model('lstm_model.h5')
scaler = joblib.load('scaler.pkl')
seq_length = 60
print("✅ Ready! No warnings.")

POPULAR = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'AMZN', 'META', 
           'NFLX', 'AMD', 'CRM', 'SPY', 'QQQ', 'BTC-USD']

def get_close_price(ticker):
    """Safe yfinance data"""
    try:
        data = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close'].iloc[:, 0]
        else:
            close = data['Close']
        return close.dropna()
    except:
        return pd.Series()

while True:
    print("\n" + "="*60)
    print("📈 STOCK PREDICTOR v2.0")
    print("Popular:", ", ".join(POPULAR))
    
    ticker = input("\nEnter ticker (quit/list): ").strip().upper()
    
    if ticker.lower() in ['quit', 'q', 'exit']:
        print("👋 Goodbye!")
        break
        
    if ticker == 'LIST':
        print("Tickers:", ", ".join(POPULAR))
        input("Press Enter...")
        continue
    
    days_input = input("Days ahead (Enter=7): ").strip()
    try:
        days = 7 if days_input == '' else int(days_input)
        if not (1 <= days <= 30):
            print("❌ 1-30 days!")
            continue
    except ValueError:
        print("❌ Number only!")
        continue
    
    data = get_close_price(ticker)
    if len(data) < seq_length:
        print(f"❌ {ticker}: insufficient data")
        input("Press Enter...")
        continue
    
    try:
        print(f"🔄 Predicting {ticker} {days} days...")
        
        # Normalize
        data_min, data_max = data.min(), data.max()
        data_norm = (data.values - data_min) / (data_max - data_min)
        seq = data_norm[-seq_length:].reshape(1, seq_length, 1)
        
        predictions = []
        for _ in range(days):
            pred = model.predict(seq, verbose=0)
            predictions.append(pred[0,0])  # FIXED: scalar
            
            # FIXED: Proper scalar assignment (no deprecation)
            new_pred = float(pred[0,0])  # Ensure scalar
            seq = np.roll(seq, -1, axis=1)
            seq[0, -1, 0] = new_pred
        
        predictions = np.array(predictions) * (data_max - data_min) + data_min
        
        # Plot
        plt.figure(figsize=(14, 8))
        recent = data.values[-120:]
        plt.plot(recent, label='Recent', linewidth=2, color='blue')
        future_x = np.arange(len(recent), len(recent)+days)
        plt.plot(future_x, predictions, 'r--', linewidth=3, label='Forecast')
        plt.title(f'{ticker}: Next {days} Days')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        latest = float(data.iloc[-1])
        print(f"\n✅ {ticker} Latest: ${latest:.2f}")
        print("Forecast:")
        for i, p in enumerate(predictions, 1):
            change = ((p - latest) / latest) * 100
            print(f"  Day {i}: ${p:.2f} ({change:+.1f}%)")
            
    except Exception as e:
        print(f"❌ {e}")
    
    input("\nPress Enter for next...")
