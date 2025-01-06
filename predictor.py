import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

# RSI hesaplama fonksiyonu
def calculate_rsi(data, window=14):
    try:
        delta = data.diff(1)
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        raise

# yfinance API'sinden veri çekme fonksiyonu
def fetch_yfinance_data(symbol='BTC-USD'):
    try:
        logger.info(f"Fetching data for {symbol}")
        df = yf.download(symbol, progress=False)  # Disable progress bar
        
        if df.empty:
            logger.error(f"No data received for symbol {symbol}")
            return pd.DataFrame()
            
        df = df[['Close', 'Volume']].copy()
        df.loc[:, 'RSI'] = calculate_rsi(df['Close'])
        df = df.dropna()
        
        logger.info(f"Successfully fetched data for {symbol}, shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

# Dataset oluşturma fonksiyonu
def create_dataset(features, target, df, look_back=3):
    dataX, dataY, dates = [], [], []
    for i in range(len(target) - look_back - 1):
        dataX.append(features[i:(i + look_back), :].flatten())
        dataY.append(target[i + look_back])
        dates.append(df.index[i + look_back])
    return np.array(dataX), np.array(dataY), dates

# Performans metriklerini hesaplama fonksiyonu
def calculate_metrics(y_true, y_pred, data_type="Test"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    accuracy = 100 - (mae / np.mean(y_true) * 100)
    return mae, rmse, r2, accuracy

# Noise ekleme fonksiyonu (rastgele tarihlere noise ekler)
def add_noise_on_random_dates(data, noise_level=0.02, noise_days=5):
    noise_data = data.copy()
    random_dates = np.random.choice(data.index, size=noise_days, replace=False)
    for event_date in random_dates:
        event_idx = data.index.get_loc(event_date)
        noise_data.iloc[event_idx] += np.random.normal(0, noise_level)
    return noise_data
