from flask import Flask, render_template, jsonify, request
import pandas as pd
import plotly.graph_objs as go
import json
from predictor import fetch_yfinance_data, create_dataset, calculate_metrics, add_noise_on_random_dates
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import numpy as np

# Gunicorn configuration
workers = 1
timeout = 120

app = Flask(__name__)

crypto_options = {
    'BTC-USD': 'Bitcoin',
    'DOGE-USD': 'Dogecoin',
    'SOL-USD': 'Solana',
    'APE-USD': 'ApeCoin'
}

@app.route('/')
def index():
    return render_template('index.html', crypto_options=crypto_options)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symbol = request.form.get('symbol')
    
    # Veri işleme
    df_daily = fetch_yfinance_data(selected_symbol).copy()  # Create a copy to avoid SettingWithCopyWarning
    scaler = MinMaxScaler()
    
    # Use loc to avoid warnings
    scaled_data = scaler.fit_transform(df_daily[['Close', 'Volume', 'RSI']])
    df_daily.loc[:, ['Close_scaled', 'Volume_scaled', 'RSI_scaled']] = scaled_data

    # Add noise using loc
    noisy_data = add_noise_on_random_dates(
        df_daily[['Close_scaled', 'Volume_scaled', 'RSI_scaled']], noise_level=0.01, noise_days=5)
    df_daily.loc[:, ['Close_scaled', 'Volume_scaled', 'RSI_scaled']] = noisy_data

    look_back = 3
    feature_columns = ['Close_scaled', 'Volume_scaled', 'RSI_scaled']
    x_daily, y_daily, dates_daily = create_dataset(
        df_daily[feature_columns].values, df_daily['Close_scaled'].values, df_daily, look_back=look_back)

    # Model eğitimi
    x_train_daily, x_test_daily, y_train_daily, y_test_daily = train_test_split(
        x_daily, y_daily, test_size=0.2, random_state=42)

    clf_daily = MLPRegressor(hidden_layer_sizes=(100, 50), early_stopping=True, 
                            max_iter=50, random_state=42, alpha=0.001, solver="adam")
    clf_daily.fit(x_train_daily, y_train_daily)

    # Tahminler
    total_pred_daily = clf_daily.predict(x_daily)
    
    # Orijinal ölçeğe dönüştürme
    total_predictions_daily = scaler.inverse_transform(
        np.hstack((total_pred_daily.reshape(-1, 1), np.zeros((len(total_pred_daily), 2)))))[:, 0]
    original_values_daily = scaler.inverse_transform(
        np.hstack((y_daily.reshape(-1, 1), np.zeros((len(y_daily), 2)))))[:, 0]

    # Gelecek tahminleri
    future_daily_dates = pd.date_range(start=pd.Timestamp.today(), periods=5, freq='D')
    last_window_daily = df_daily[feature_columns].values[-look_back:]
    future_predictions_daily = []

    for _ in range(len(future_daily_dates)):
        next_pred_daily = clf_daily.predict(last_window_daily.flatten().reshape(1, -1))[0]
        future_predictions_daily.append(next_pred_daily)
        last_window_daily = np.roll(last_window_daily, -1, axis=0)
        last_window_daily[-1, 0] = next_pred_daily

    future_predictions_daily = scaler.inverse_transform(
        np.hstack((np.array(future_predictions_daily).reshape(-1, 1), 
                  np.zeros((len(future_predictions_daily), 2)))))[:, 0]

    # Grafik verilerini hazırlama
    historical_data = {
        'x': [d.strftime('%Y-%m-%d') for d in dates_daily],
        'y': original_values_daily.tolist(),
        'name': 'Historical Data',
        'mode': 'lines',
        'type': 'scatter'
    }

    prediction_data = {
        'x': [d.strftime('%Y-%m-%d') for d in dates_daily],
        'y': total_predictions_daily.tolist(),
        'name': 'Predictions',
        'mode': 'lines',
        'type': 'scatter'
    }

    future_data = {
        'x': [d.strftime('%Y-%m-%d') for d in future_daily_dates],
        'y': future_predictions_daily.tolist(),
        'name': 'Future Predictions',
        'mode': 'lines+markers',
        'type': 'scatter'
    }

    rsi_data = {
        'x': df_daily.index.strftime('%Y-%m-%d').tolist(),
        'y': df_daily['RSI'].tolist(),
        'name': 'RSI',
        'yaxis': 'y2',
        'type': 'scatter'
    }

    # Performans metrikleri
    train_metrics = calculate_metrics(y_train_daily, clf_daily.predict(x_train_daily), "Train")
    test_metrics = calculate_metrics(y_test_daily, clf_daily.predict(x_test_daily), "Test")

    return jsonify({
        'historical_data': historical_data,
        'prediction_data': prediction_data,
        'future_data': future_data,
        'rsi_data': rsi_data,
        'train_metrics': {
            'mae': float(train_metrics[0]),
            'rmse': float(train_metrics[1]),
            'r2': float(train_metrics[2]),
            'accuracy': float(train_metrics[3])
        },
        'test_metrics': {
            'mae': float(test_metrics[0]),
            'rmse': float(test_metrics[1]),
            'r2': float(test_metrics[2]),
            'accuracy': float(test_metrics[3])
        },
        'future_predictions': [
            {'date': date.strftime('%Y-%m-%d'), 'price': float(price)} 
            for date, price in zip(future_daily_dates, future_predictions_daily)
        ]
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port) 