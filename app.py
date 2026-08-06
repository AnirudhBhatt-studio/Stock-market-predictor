import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging warnings

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# NLTK for VADER Sentiment Analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

@st.cache_resource
def init_sentiment_analyzer():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = init_sentiment_analyzer()

# ================= PAGE CONFIG =================
st.set_page_config(layout="wide", page_title="Accelerated Stock Predictor", page_icon="🚀")

# ================= HAIKEI & MOTION STYLING =================
st.markdown("""
<style>
    /* Haikei-inspired Layered Wave Background */
    .stApp {
        background-color: #0e1117;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 800 500"><defs><linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:%231a1c2e;stop-opacity:1" /><stop offset="100%" style="stop-color:%230e1117;stop-opacity:1" /></linearGradient></defs><rect width="100%" height="100%" fill="url(%23grad)"/><path d="M0,192L48,202.7C96,213,192,235,288,224C384,213,480,171,576,165.3C672,160,768,192,816,208L864,224L864,500L816,500C768,500,672,500,576,500C480,500,384,500,288,500C192,500,96,500,48,500L0,500Z" fill="%23161b26" fill-opacity="0.6"></path><path d="M0,320L48,304C96,288,192,256,288,261.3C384,267,480,309,576,304C672,299,768,245,816,218.7L864,192L864,500L816,500C768,500,672,500,576,500C480,500,384,500,288,500C192,500,96,500,48,500L0,500Z" fill="%23212838" fill-opacity="0.4"></path></svg>');
        background-size: cover;
        background-attachment: fixed;
    }

    /* Glassmorphism Card Containers */
    div[data-testid="stMetricValue"], div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 16px 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 210, 255, 0.4);
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .hero-subtitle {
        color: #8a99ad;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(14, 17, 23, 0.85);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# ================= NAVIGATION & SIDEBAR =================
st.sidebar.title("Navigation")
nav_choice = st.sidebar.radio("Go to", ["Summary & Forecast", "News & Sentiment", "Statistics & Analysis"])

st.sidebar.subheader("Model Config")
stock = st.sidebar.text_input("Ticker Symbol", "AAPL").upper().strip()
prediction_days = st.sidebar.slider("Days to Predict", 1, 30, 10)
epochs = st.sidebar.number_input("Training Epochs", 10, 200, 30)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
training_years = st.sidebar.slider("Training Data Years", 1, 10, 5)
use_all_data = st.sidebar.checkbox("Use all available historical data", value=False)

# ================= DATA LOADING =================
@st.cache_data
def load_data(ticker, years, use_all):
    period = "max" if use_all else f"{years}y"
    data = yf.download(ticker, period=period)
    if data.empty:
        return None
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.index = pd.to_datetime(data.index).normalize()
    data = data.ffill().dropna()
    return data

# ================= DEEP LEARNING MODEL =================
@st.cache_resource
def train_deep_model(data, epochs, batch_size):
    close_prices = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    return model, scaler, scaled_data

# ================= PREDICTION LOGIC =================
def predict_future(model, scaler, last_60_days, days_to_predict):
    current_batch = last_60_days.reshape(1, 60, 1)
    future_predictions = []

    for _ in range(days_to_predict):
        pred = model.predict(current_batch, verbose=0)
        future_predictions.append(pred[0])
        current_batch = np.append(current_batch[:, 1:, :], [pred], axis=1)

    return scaler.inverse_transform(future_predictions)

df = load_data(stock, training_years, use_all_data)

# ================= VIEW 1: SUMMARY & FORECAST =================
if nav_choice == "Summary & Forecast":
    st.markdown('<p class="hero-title">🚀 Accelerated Stock Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Deep LSTM Neural Network with dynamic Plotly auto-scaling</p>', unsafe_allow_html=True)

    if df is not None:
        latest_price = float(df['Close'].iloc[-1])
        
        col1, col2 = st.columns(2)
        col1.metric("Active Ticker", stock)
        col2.metric("Latest Close Price", f"${latest_price:,.2f}")

        st.link_button(
            f"🔍 Is {stock} a Long-Term Buy? View Analysis on Yahoo Finance",
            f"https://finance.yahoo.com/quote/{stock}/analysis"
        )

        st.write("---")

        with st.status("Training Neural Network...", expanded=True) as status:
            st.write("Preprocessing historical time-series sequences...")
            model, scaler, scaled_data = train_deep_model(df, epochs, batch_size)
            st.write("Generating future price vectors...")
            future_prices = predict_future(model, scaler, scaled_data[-60:], prediction_days)
            status.update(label="Training Complete!", state="complete", expanded=False)

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index[-200:],
            y=df['Close'].iloc[-200:].values.flatten(),
            name="Recent History",
            line=dict(color='#00d2ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 210, 255, 0.05)'
        ))

        connect_x = [df.index[-1]] + list(future_dates)
        connect_y = [latest_price] + list(future_prices.flatten())

        fig.add_trace(go.Scatter(
            x=connect_x,
            y=connect_y,
            name="LSTM Forecast",
            line=dict(color='#ff4b4b', width=3, dash='dash')
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500,
            hovermode="x unified",
            autosize=True,
            margin=dict(l=20, r=20, t=30, b=20)
        )

        fig.update_yaxes(autorange=True, fixedrange=False)
        fig.update_xaxes(autorange=True, fixedrange=False)

        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Next {prediction_days}-Day Forecast Model")
        res_df = pd.DataFrame({"Date": future_dates.date, "Projected Price": future_prices.flatten()})
        st.dataframe(res_df.style.format({"Projected Price": "${:,.2f}"}), use_container_width=True)

    else:
        st.error(f"Could not load data for symbol '{stock}'. Please verify the ticker symbol.")

# ================= VIEW 2: NEWS & SENTIMENT =================
elif nav_choice == "News & Sentiment":
    st.markdown(f'<p class="hero-title">📰 News & Sentiment Analysis — {stock}</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Real-time Yahoo Finance headlines processed via VADER Sentiment Lexicon</p>', unsafe_allow_html=True)

    ticker_obj = yf.Ticker(stock)
    news_items = ticker_obj.news

    if news_items:
        parsed_news = []
        for item in news_items:
            # Handle variations in yfinance API return structures
            content = item.get('content', {})
            title = content.get('title') or item.get('title', 'No Title')
            provider = content.get('provider', {}).get('displayName') or item.get('publisher', 'Unknown Source')
            link = content.get('canonicalUrl', {}).get('url') or item.get('link', '#')

            # VADER compound score computation
            scores = sia.polarity_scores(title)
            compound = scores['compound']
            
            if compound >= 0.05:
                sentiment = "Positive"
            elif compound <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            parsed_news.append({
                "Title": title,
                "Source": provider,
                "Sentiment": sentiment,
                "Compound Score": compound,
                "Link": link
            })

        news_df = pd.DataFrame(parsed_news)

        avg_compound = news_df["Compound Score"].mean()
        col1, col2 = st.columns(2)
        
        col1.metric("Overall Sentiment Score", f"{avg_compound:.2f}")
        if avg_compound >= 0.05:
            col2.success("Market Outlook: Bullish / Positive Sentiment")
        elif avg_compound <= -0.05:
            col2.error("Market Outlook: Bearish / Negative Sentiment")
        else:
            col2.info("Market Outlook: Neutral Sentiment")

        st.write("---")
        st.subheader("Recent Headlines")

        for _, row in news_df.iterrows():
            badge_color = "🟢" if row['Sentiment'] == "Positive" else ("🔴" if row['Sentiment'] == "Negative" else "⚪")
            st.markdown(f"### {badge_color} [{row['Title']}]({row['Link']})")
            st.caption(f"Source: **{row['Source']}** | Sentiment Score: **{row['Compound Score']:.2f}** ({row['Sentiment']})")
            st.write("---")
    else:
        st.warning(f"No recent news articles were retrieved for {stock}.")

# ================= VIEW 3: STATISTICS & ANALYSIS =================
elif nav_choice == "Statistics & Analysis":
    st.markdown(f'<p class="hero-title">📊 Valuation & Statistics — {stock}</p>', unsafe_allow_html=True)
    
    st.link_button(
        f"🔗 Open {stock} directly on Yahoo Finance",
        f"https://finance.yahoo.com/quote/{stock}"
    )

    st.write("---")

    ticker_obj = yf.Ticker(stock)
    
    try:
        info = ticker_obj.info
        company_name = info.get('longName', stock)
        trailing_pe = info.get('trailingPE', 'N/A')
        forward_pe = info.get('forwardPE', 'N/A')
        market_cap = info.get('marketCap', None)
        profit_margins = info.get('profitMargins', None)

        mcap_str = f"${market_cap / 1e9:,.2f}B" if market_cap else "N/A"
        margin_str = f"{profit_margins * 100:.2f}%" if profit_margins else "N/A"

        st.info(
            f"**{company_name} ({stock})** current Market Cap stands at **{mcap_str}** "
            f"with a Trailing P/E of **{trailing_pe}** and Forward P/E of **{forward_pe}**. "
            f"Profit margin is currently recorded at **{margin_str}**."
        )

        st.subheader("Valuation Measures")

        metrics_df = pd.DataFrame({
            "Metric": [
                "Market Cap", 
                "Enterprise Value", 
                "Trailing P/E", 
                "Forward P/E", 
                "PEG Ratio (5yr expected)", 
                "Price/Sales", 
                "Price/Book"
            ],
            "Current Value": [
                mcap_str,
                f"${info.get('enterpriseValue', 0) / 1e9:,.2f}B" if info.get('enterpriseValue') else "N/A",
                trailing_pe,
                forward_pe,
                info.get('pegRatio', 'N/A'),
                info.get('priceToSalesTrailing12Months', 'N/A'),
                info.get('priceToBook', 'N/A')
            ]
        })

        st.dataframe(metrics_df, use_container_width=True)

    except Exception:
        st.warning(f"Could not load detailed live valuation metadata for '{stock}'. Check ticker validity.")

    st.write("---")
    st.link_button(
        f"🔍 Is {stock} a Long-Term Buy? View Analyst Consensus on Yahoo Finance",
        f"https://finance.yahoo.com/quote/{stock}/analysis"
    )
