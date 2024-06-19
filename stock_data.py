import streamlit as st
from datetime import datetime
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import talib as ta
from pygooglenews import GoogleNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from transformers import pipeline
from newspaper import Article

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
    data = yf.download(stock, START, TODAY)
    data.reset_index(inplace=True)
    return data

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_candlestick(data, title):
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'],
                                         name='Candlesticks')])
    fig.layout.update(title_text=title, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_technical_indicators(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], name='RSI'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA'], name='SMA'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA'], name='EMA'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], name='MACD'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Signal'], name='Signal'))
    fig.layout.update(title_text="Technical Indicators", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

def fetch_full_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return "Unable to fetch article."

def recognize_pattern(data):
    patterns = {
        "CDLHAMMER": ta.CDLHAMMER,
        "CDLINVERTEDHAMMER": ta.CDLINVERTEDHAMMER,
        "CDLHANGINGMAN": ta.CDLHANGINGMAN,
        "CDLSHOOTINGSTAR": ta.CDLSHOOTINGSTAR,
        "CDLENGULFING": ta.CDLENGULFING,
        "CDLPIERCING": ta.CDLPIERCING,
        "CDLDOJI": ta.CDLDOJI,
        "CDL3WHITESOLDIERS": ta.CDL3WHITESOLDIERS,
        "CDL3BLACKCROWS": ta.CDL3BLACKCROWS,
        "CDLMORNINGSTAR": ta.CDLMORNINGSTAR,
        "CDLEVENINGSTAR": ta.CDLEVENINGSTAR
    }
    
    pattern_result = {"bullish": [], "bearish": []}

    for pattern_name, pattern_func in patterns.items():
        result = pattern_func(data['Open'], data['High'], data['Low'], data['Close'])
        if result[result != 0].any():
            if result[result > 0].any():
                pattern_result["bullish"].append(pattern_name)
            if result[result < 0].any():
                pattern_result["bearish"].append(pattern_name)

    return pattern_result

def main():
    st.title("Stock Information")
    symbol = st.text_input("Enter stock symbol:", key="symbol_input")
    n_years = st.slider("Years of prediction:", 1, 5)
    period = n_years * 365
    
    if symbol:
        if 'last_symbol' not in st.session_state:
            st.session_state.last_symbol = ""
        if symbol != st.session_state.last_symbol:
            st.session_state.last_symbol = symbol
            st.empty()  # Clear previous plots
            
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
            
            st.write("Technical Analysis")
            data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
            data['SMA'] = ta.SMA(data['Close'], timeperiod=20)
            data['EMA'] = ta.EMA(data['Close'], timeperiod=20)
            data['MACD'], data['Signal'], data['Histogram'] = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            st.write(data.tail())

            plot_technical_indicators(data)
            
            st.write("Pattern Recognition")
            pattern_result = recognize_pattern(data)
            if pattern_result["bullish"]:
                st.write("Bullish Patterns Detected:")
                st.write("Number of Bullish Patterns:", len(pattern_result["bullish"]))
                plot_candlestick(data, "Bullish Candlestick Patterns")
            if pattern_result["bearish"]:
                st.write("Bearish Patterns Detected:")
                st.write("Number of Bearish Patterns:", len(pattern_result["bearish"]))
                plot_candlestick(data, "Bearish Candlestick Patterns")
            else:
                st.write("No significant pattern detected.")
            
            stock = search_stock(symbol)
        if stock:
            st.write("News")
            googlenews = GoogleNews()
            news = googlenews.search(f"{symbol} stock price OR market OR performance OR earnings OR financial OR news", when='1d')
            articles = news['entries']
            
            st.write(f"Total Number of Articles: {len(articles)}")  # Total number of articles
            
            # Initialize sentiment analysis pipeline
            # sentiment_analysis = pipeline("sentiment-analysis")
            
            # sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
            
            # for item in articles:
            #     st.write("Title:", item['title'])
            #     st.write("Link:", item['link'])
            #     st.write("Published:", item['published'])

            #     # Fetch full article content
            #     full_content = fetch_full_article(item['link'])
            #     st.write("Content:", full_content[:500])  # Display first 500 characters of the content

            #     # Analyze sentiment of the full content using Hugging Face
            #     sentiment_score = sentiment_analysis(full_content[:512])[0]  # Truncate to first 512 characters for analysis
            #     st.write("Sentiment:", sentiment_score['label'], "-", sentiment_score['score'])
                
            #     # Count sentiment
            #     if sentiment_score['label'] == 'POSITIVE':
            #         sentiments['positive'] += 1
            #     elif sentiment_score['label'] == 'NEGATIVE':
            #         sentiments['negative'] += 1
            #     else:
            #         sentiments['neutral'] += 1
                
            #     st.write("")

            # # Plot sentiment distribution as a pie chart
            # fig, ax = plt.subplots()
            # ax.pie(sentiments.values(), labels=sentiments.keys(), autopct='%1.1f%%', colors=['green', 'blue', 'red'])
            # ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            # st.pyplot(fig)

            vader_analyzer = SentimentIntensityAnalyzer()
            
            sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
            
            for item in articles:
                st.write("Title:", item['title'])
                st.write("Link:", item['link'])
                st.write("Published:", item['published'])

                # Fetch full article content
                full_content = fetch_full_article(item['link'])
                st.write("Content:", full_content[:500])  # Display first 500 characters of the content

                # Analyze sentiment of the full content using Vader
                sentiment_score = vader_analyzer.polarity_scores(full_content)
                st.write("Sentiment:", sentiment_score)

                # Count sentiment
                if sentiment_score['compound'] >= 0.70:
                    sentiments['positive'] += 1
                elif sentiment_score['compound'] < 0.00:
                    sentiments['negative'] += 1
                else:
                    sentiments['neutral'] += 1
                
                st.write("")

            # Plot sentiment distribution as a pie chart
            fig, ax = plt.subplots()
            ax.pie(sentiments.values(), labels=sentiments.keys(), autopct='%1.1f%%', colors=['green', 'blue', 'red'])
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)

            
        else:
            st.write("Stock not found.")

if __name__ == "__main__":
    main()
