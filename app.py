from flask import Flask, request, render_template, jsonify
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
from newspaper import Article
from pyngrok import ngrok

app = Flask(__name__)

START = "2015-01-01"
TODAY = datetime.today().strftime("%Y-%m-%d")

def search_stock(symbol):
    try:
        stock = yf.Ticker(symbol)
        if 'longName' in stock.info:
            return stock
        else:
            return None
    except Exception as e:
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
    return fig.to_html(full_html=False)

def plot_candlestick(data, title):
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'],
                                         name='Candlesticks')])
    fig.layout.update(title_text=title, xaxis_rangeslider_visible=True)
    return fig.to_html(full_html=False)

def plot_technical_indicators(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], name='RSI'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA'], name='SMA'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA'], name='EMA'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], name='MACD'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Signal'], name='Signal'))
    fig.layout.update(title_text="Technical Indicators", xaxis_rangeslider_visible=True)
    return fig.to_html(full_html=False)

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/stock", methods=["POST"])
def stock():
    symbol = request.form.get("symbol")
    n_years = int(request.form.get("n_years", 1))
    period = n_years * 365

    stock = search_stock(symbol)
    if stock:
        data = load_data(symbol)
        raw_data_plot = plot_raw_data(data)

        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        model = Prophet()
        model.fit(df_train)
        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)

        forecast_plot = plot_plotly(model, forecast).to_html(full_html=False)

        data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
        data['SMA'] = ta.SMA(data['Close'], timeperiod=20)
        data['EMA'] = ta.EMA(data['Close'], timeperiod=20)
        data['MACD'], data['Signal'], data['Histogram'] = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        technical_indicators_plot = plot_technical_indicators(data)

        pattern_result = recognize_pattern(data)
        if pattern_result["bullish"] or pattern_result["bearish"]:
            candlestick_plot = plot_candlestick(data, "Candlestick Patterns")
        else:
            candlestick_plot = "No significant pattern detected."

        googlenews = GoogleNews()
        news = googlenews.search(f"{symbol} stock price OR market OR performance OR earnings OR financial OR news", when='1d')
        articles = news['entries']

        vader_analyzer = SentimentIntensityAnalyzer()
        sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
        articles_data = []

        for item in articles:
            full_content = fetch_full_article(item['link'])
            sentiment_score = vader_analyzer.polarity_scores(full_content)

            if sentiment_score['compound'] >= 0.70:
                sentiments['positive'] += 1
            elif sentiment_score['compound'] < 0.00:
                sentiments['negative'] += 1
            else:
                sentiments['neutral'] += 1

            articles_data.append({
                "title": item['title'],
                "link": item['link'],
                "published": item['published'],
                "content": full_content[:500],
                "sentiment": sentiment_score
            })

        fig, ax = plt.subplots()
        if sum(sentiments.values()) > 0:  # Check if sentiments have valid values
            ax.pie(sentiments.values(), labels=sentiments.keys(), autopct='%1.1f%%', colors=['green', 'blue', 'red'])
            ax.axis('equal')
            sentiment_pie_chart = plt_to_html(fig)
        else:
            sentiment_pie_chart = "No sentiment data available."

        return jsonify({
            "raw_data_plot": raw_data_plot,
            "forecast_plot": forecast_plot,
            "technical_indicators_plot": technical_indicators_plot,
            "candlestick_plot": candlestick_plot,
            "articles_data": articles_data,
            "sentiment_pie_chart": sentiment_pie_chart
        })
    else:
        return jsonify({"error": "Stock not found."})

def plt_to_html(fig):
    import io
    from base64 import b64encode
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = b64encode(buf.read()).decode("utf-8")
    buf.close()
    return f'<img src="data:image/png;base64,{image_base64}"/>'

if __name__ == "__main__":
    port = 5000
    public_url = ngrok.connect(port)
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
    app.run(port=port)
