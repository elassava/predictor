# 🚀 Cryptocurrency Price Prediction Using Historical Data

## 📊 Overview

The main purpose of this project, **"Predicting Cryptocurrency Price Movements by Using Historical Data"**, is to predict and foresee the behavior of specific cryptocurrencies in the near future by leveraging their historical data. This project establishes a connection between the rapidly evolving fields of **Finance** and **Machine Learning**, contributing to the understanding of market trends and supporting informed decision-making.

The project utilizes **Machine Learning techniques**, specifically **Multi-Layer Perceptron (MLP) Regressor**, to forecast daily cryptocurrency prices and analyze their trends. By integrating historical data, key financial indicators, and advanced modeling techniques, the project aims to provide a robust and interpretable prediction framework.

---

## ✨ Features and Objectives

### 🎯 Key Features

1. **Technical Indicator - RSI Calculation**:
   * The **Relative Strength Index (RSI)**, a momentum-based technical indicator, is calculated to measure market conditions (e.g., overbought or oversold).

2. **Data Preprocessing and Normalization**:
   * Historical data is normalized using **MinMaxScaler** to scale features like closing price, trading volume, and RSI values between 0 and 1, improving model convergence.
   * A **sliding window approach** is implemented to generate lag-based feature sets for time-series prediction.

3. **Prediction and Visualization**:
   * Historical predictions are compared to actual values to evaluate model accuracy.
   * Future price predictions are visualized for selected cryptocurrencies.
   * Bar charts highlight prediction accuracy across training, testing, and future datasets.
   * Percentage change analysis compares historical and predicted trends.

4. **Dynamic Financial Data Retrieval**:
   * Data is fetched from **Yahoo Finance API** for cryptocurrencies like Bitcoin (BTC), Dogecoin (DOGE), Solana (SOL), and ApeCoin (APE).

### 🎯 Objectives

* Predict cryptocurrency price movements with high accuracy using historical data.
* Assist investors in managing risks, identifying profit opportunities, and preparing for market uncertainties.
* Explore the intersection of **Machine Learning** and **Financial Technologies** to enhance cryptocurrency analysis.

---

## 🔍 Methodology

### 📥 Data Collection

* Historical data is collected from Yahoo Finance for four cryptocurrencies:
  * **Bitcoin (BTC)**, **Dogecoin (DOGE)**, **Solana (SOL)**, and **ApeCoin (APE)**.
* Features include:
  * **Closing Price**: Reflects market trends and sentiment.
  * **Volume**: Represents market activity and liquidity.
  * **RSI (Relative Strength Index)**: Indicates momentum by identifying overbought/oversold conditions.
* Data from the past 8 years is utilized, split into **80% training** and **20% testing** subsets using `train_test_split`.

---

## 🛠️ Tools and Libraries

The following Python libraries and tools were used:

* **Pandas**: Data manipulation and analysis (loading, cleaning, and transforming cryptocurrency data).
* **NumPy**: Numerical computations for feature preparation, percentage change calculations, and array operations.
* **Matplotlib**: Visualization of results, including actual vs. predicted prices and accuracy comparisons.
* **scikit-learn**:
  * `MinMaxScaler` for feature normalization.
  * `train_test_split` for data splitting.
  * `MLPRegressor` for time-series prediction.
  * Evaluation metrics like MAE, RMSE, and R².
* **yfinance**: Retrieval of historical financial data (e.g., daily closing prices, trading volumes).

---

## 📈 Results and Visualization

---

## ⚠️ Limitations and Future Work

### Limitations

1. External market factors like regulatory changes or societal events are not accounted for, which can lead to deviations in forecasts.
2. Predictions are sensitive to the chosen look-back period and feature set.
3. Only a limited set of technical indicators (Close, Volume, RSI) is used.

### 🔮 Future Improvements

1. Incorporate additional technical indicators like moving averages (e.g., SMA, EMA) and **MACD** for enhanced predictions.
2. Perform sentiment analysis using social media data to capture public perception and its impact on cryptocurrency prices.
3. Utilize **Deep Learning models** (e.g., LSTM, GRU) for improved time-series forecasting.

---

## 🚀 How to Use

### Prerequisites

* Python 3.8 or above
* Install the mentioned dependencies 

### Running the Project

1. Select a cryptocurrency to analyze (BTC, DOGE, SOL, APE).
2. The script will fetch historical data and perform preprocessing.
3. Train the MLP model using the processed dataset.
4. View historical price predictions and future price forecasts.
5. Analyze prediction accuracy and percentage changes.

### Output

* Historical and predicted price charts.
* Accuracy bar charts for train, test, and future predictions.
* Percentage change comparison graphs.

---

## 👥 Authors

* **Ali Zekai Deveci**
* **Ela Semra Sava**
* **Münib Akar**

This project was developed as part of the course **SE 3007 | Introduction to Machine Learning**.

---

## 🙏 Acknowledgments

* **Yahoo Finance**: For providing historical cryptocurrency data.
* **scikit-learn**: For offering tools to implement and evaluate the MLP Regressor. 