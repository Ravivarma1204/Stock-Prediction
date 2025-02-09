from flask import Flask, render_template, request
import yfinance as yf
from keras.models import load_model
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the LSTM model
MODEL_PATH = 'models/Stock_Predictions_Model_GOOG_2022-12-21.h5'
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Function to calculate moving average (50-day moving average)
def calculate_moving_average(stock_symbol, days=50):
    stock_data = yf.Ticker(stock_symbol)
    hist = stock_data.history(period="1y")
    moving_avg = hist['Close'].rolling(window=days).mean().iloc[-1]  # Latest moving average
    return moving_avg

# Function to get stock recommendation
def get_recommendation(stock_symbol):
    recommendation = "Buy"  # Example: replace with actual recommendation logic
    confidence_score = 90  # Example: replace with actual confidence logic
    return recommendation, confidence_score

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stock_prediction', methods=['POST'])
def stock_prediction():
    stock_symbol = request.form['stock_symbol']
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)

    # Fetch stock data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    if stock_data.empty:
        return render_template('index.html', error="Invalid stock name or no data available.")

    # Prepare data for prediction
    stock_data['Return'] = stock_data['Close'].pct_change()
    data = stock_data[['Close']].dropna()
    scaled_data = (data - data.mean()) / data.std()
    input_data = scaled_data[-60:].values.reshape(1, 60, 1)

    # Make prediction
    prediction = model.predict(input_data)
    predicted_price = prediction[0][0] * data.std().values[0] + data.mean().values[0]

    # Generate the prediction graph
    fig, ax = plt.subplots()
    ax.plot(stock_data.index, stock_data['Close'], label='Actual Price')
    ax.axvline(x=stock_data.index[-1], color='red', linestyle='--', label='Prediction')
    ax.plot(stock_data.index[-1], predicted_price, 'go', label='Predicted Price')

    ax.set_title(f"{stock_symbol} - Stock Price Prediction")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()

    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('stock_prediction.html', stock_name=stock_symbol, predicted_price=predicted_price, plot_url=plot_url)

@app.route('/stock_details', methods=['POST'])
def stock_details():
    stock_symbol = request.form['stock_symbol']
    action = request.form['action']
    
    if action == 'details':
        # Fetch stock details from Yahoo Finance
        try:
            stock_data = yf.Ticker(stock_symbol)
            stock_info = stock_data.info  # Get general info about the stock

            if not stock_info:
                return render_template('index.html', error="No data found for the stock symbol.")
            
            details = {
                'Symbol': stock_info.get('symbol', 'N/A'),
                'Name': stock_info.get('longName', 'N/A'),
                'Sector': stock_info.get('sector', 'N/A'),
                'Industry': stock_info.get('industry', 'N/A'),
                'Market Cap': stock_info.get('marketCap', 'Not Available'),
                'PE Ratio': stock_info.get('trailingPE', 'Not Available'),
                'EPS': stock_info.get('trailingEps', 'Not Available'),
                'Dividend Yield': stock_info.get('dividendYield', 'Not Available'),
                '52 Week High': stock_info.get('fiftyTwoWeekHigh', 'Not Available'),
                '52 Week Low': stock_info.get('fiftyTwoWeekLow', 'Not Available'),
                'Previous Close': stock_info.get('regularMarketPreviousClose', 'Not Available'),
                'Return on Equity (ROE)': stock_info.get('returnOnEquity', 'Not Available'),
                'Debt': stock_info.get('totalDebt', 'Not Available'),
                'Sector PE': stock_info.get('sectorPe', 'Not Available'),
            }
            return render_template('stock_details.html', stock_name=stock_symbol, details=details)

        except Exception as e:
            error_message = f"Error fetching details for {stock_symbol}: {str(e)}"
            return render_template('index.html', error=error_message)

    elif action == 'prediction':
        # Prediction logic
        return stock_prediction()  # Call the existing prediction function

    elif action == 'recommendation':
        # Recommendation logic
        return stock_recommendation(stock_symbol)  # Call the existing recommendation function

    else:
        return render_template('index.html', error="Invalid action.")

def stock_recommendation(stock_symbol):
    # Your logic to fetch stock data and generate recommendations
    try:
        stock_data = yf.Ticker(stock_symbol)
        stock_info = stock_data.info
        
        # Fetch relevant stock data
        last_close = stock_info.get('regularMarketPreviousClose', 'Not Available')
        moving_avg = calculate_moving_average(stock_symbol)  # Example function for moving average
        recommendation, confidence_score = get_recommendation(stock_symbol)  # Your recommendation logic
        
        return render_template('stock_recommendation.html', 
                               stock_symbol=stock_symbol, 
                               last_close=last_close, 
                               moving_avg=moving_avg, 
                               recommendation=recommendation, 
                               confidence_score=confidence_score)
    except Exception as e:
        error_message = f"Error fetching recommendation for {stock_symbol}: {str(e)}"
        return render_template('stock_recommendation.html', error=error_message)

def classify_by_market_cap(market_cap):
    if market_cap is None:
        return "Unknown"
    elif market_cap > 10**10:
        return "Large Cap"
    elif market_cap > 2*10**9:
        return "Mid Cap"
    else:
        return "Small Cap"

if __name__ == '__main__':
    app.run(debug=True)
