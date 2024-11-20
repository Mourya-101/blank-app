import streamlit as st
import yfinance as yf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import openai  # Make sure to install openai with `pip install openai`

# Sidebar: App title and market selection
st.sidebar.title("Premium Stock Investment Dashboard")

# Dropdown for selecting the market region
market_options = ["Global Market", "Indian Market", "American Market", "European Market", "Asian Market"]
region = st.sidebar.selectbox("Select Market Region", market_options)
st.sidebar.write(f"Explore the top stocks for: {region}")

# Define top stocks based on selected region
top_stocks = {
    "Global Market": ["TSLA", "AMZN", "BABA"],
    "Indian Market": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
    "American Market": ["AAPL", "GOOGL", "MSFT"],
    "European Market": ["SAP.DE", "AIR.PA", "RDSA.AS"],
    "Asian Market": ["SONY.T", "SMFG.T", "NTTYY"]
}

# Option for users to select from predefined stocks or enter a custom ticker
stock_input_method = st.sidebar.radio("Choose stock selection method", ["Select from list", "Enter ticker manually"])

if stock_input_method == "Select from list":
    # Display top stocks based on selected region
    selected_stock = st.sidebar.selectbox("Choose a Stock", top_stocks[region])
else:
    # Allow user to enter a custom stock ticker
    selected_stock = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()

# Display selected region and stock
st.title(f"Stock Analysis for {selected_stock} in {region}")

# Function to retrieve stock data
@st.cache_data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="10y")
    return data

# Retrieve stock data and display current price
try:
    data = get_stock_data(selected_stock)
    current_price = data['Close'].iloc[-1]
    st.write(f"**Current Stock Price for {selected_stock}:** ${current_price:.2f}")

    # Plot historical data
    st.subheader("Historical Price")
    st.line_chart(data['Close'])

    # Data preparation for the 5-year forecast
    def prepare_data(data, lookback=60):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    # Define and train LSTM model
    def train_lstm_model(data, epochs=10):
        X, y = prepare_data(data.values)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=3)])

        return model

    # Train model on historical data
    model = train_lstm_model(data['Close'])

    # Forecasting future prices
    def forecast_5_years(model, last_data, steps=1250):
        future = []
        current_input = last_data[-60:].reshape(1, 60, 1)  # Ensure it's in (1, 60, 1) shape

        for _ in range(steps):
            prediction = model.predict(current_input, verbose=0)
            future.append(prediction[0, 0])
            current_input = np.append(current_input[:, 1:, :], [[prediction[0]]], axis=1)  # Keep (1, 60, 1) shape

        return future

    # Generate and display forecasted stock prices
    st.subheader("5-Year Stock Price Forecast")
    forecasted_prices = forecast_5_years(model, data['Close'].values)

    # Display future prices as a line chart
    fig, ax = plt.subplots()
    ax.plot(range(len(forecasted_prices)), forecasted_prices, label="5-Year Forecast", color="orange")
    ax.set_xlabel("Days")
    ax.set_ylabel("Forecasted Price")
    st.pyplot(fig)

    # Set up OpenAI API for LLM-based recommendation
    openai.api_key = "your_openai_api_key"  # Replace with your OpenAI API key

    # Define recommendation function
    def get_investment_recommendation(prices, ticker):
        prompt = f"The 5-year forecast for stock {ticker} shows prices trending as follows: {prices[:5]}... \
        Provide a recommendation on whether one should invest in this stock, considering market trends, \
        volatility, and long-term potential."

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()

    # Display the recommendation
    recommendation = get_investment_recommendation(forecasted_prices, selected_stock)
    st.subheader("Investment Recommendation")
    st.write(recommendation)

except Exception as e:
    st.error(f"An error occurred while retrieving the stock data: {e}")

# Additional Tabs for Commodity Analysis
tab = st.sidebar.radio("Analyze Commodities", ["Gold", "Oil", "Cryptocurrency"])

st.header(f"{tab} Analysis")
if tab == "Gold":
    st.write("Gold price trends, supply-demand analysis, and more insights...")
elif tab == "Oil":
    st.write("Oil price analysis, volatility trends, and factors affecting prices...")
elif tab == "Cryptocurrency":
    st.write("Crypto market trends, volatility, and top-performing cryptos...")
