import streamlit as st
from datetime import datetime
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START= "2015-01-01"
TODAY= datetime.today().strftime("%Y-%m-%d")

st.title("MY FINANCE APP")

def search_stock(symbol):
    try:
        stock = yf.Ticker(symbol)
        if 'longName' in stock.info:
            return stock
        else:
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def load_data(stock):
    data=yf.download(stock, START, TODAY)
    data.reset_index(inplace=True)
    return data

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def main():
    st.title("Stock Information")
    symbol = st.text_input("Enter stock symbol:")
    n_years = st.slider("Years of prediction:", 1, 5)
    period = n_years * 365
    
    if symbol:
        stock = search_stock(symbol)
        if stock:
            st.write("Stock found:")
            data_load_state = st.text("Loading data...")
            data = load_data(symbol)
            data_load_state.text("Data Loaded Successfully!")
            st.subheader("Raw Data")
            st.write(data.tail())
            plot_raw_data(data)
            df_train = data[['Date', 'Close']]
            df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})
            model = Prophet()
            model.fit(df_train)  
            future = model.make_future_dataframe(periods=period)
            forecast = model.predict(future)

            st.subheader("Forecast Data")
            st.write(forecast.tail())


            st.write("Forecast Data")
            fig1 = plot_plotly(model, forecast)
            st.plotly_chart(fig1)

            st.write("Forecast Components")
            fig2 = model.plot_components(forecast)
            st.write(fig2)
        else:
            st.write("Stock not found.")



if __name__ == "__main__":
    main()

