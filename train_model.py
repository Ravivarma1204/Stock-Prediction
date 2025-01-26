import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# List of Indian stock symbols (use ".NS" for NSE listed stocks)
stock_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS', 'SBIN.NS']  # Example symbols for Indian stocks

# Function to fetch stock data
def fetch_data(symbols, start_date, end_date):
    data = {}
    for symbol in symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        stock_data['Symbol'] = symbol  # Add stock symbol as a feature
        data[symbol] = stock_data
    return data

# Fetch stock data for multiple Indian stocks
start_date = '2015-01-01'
end_date = '2022-01-01'
data = fetch_data(stock_symbols, start_date, end_date)

# Combine all stock data into a single DataFrame
combined_data = pd.concat(data.values())
combined_data.reset_index(drop=True, inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(combined_data['Close'].values.reshape(-1, 1))

# Prepare training data
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Reshape data for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Save the model
model.save('models/Stock_Predictions_Model_Indian_2022-12-21.h5')
print("Model saved successfully.")
