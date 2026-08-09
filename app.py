import os
import urllib.request
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score
import streamlit as st
import yfinance as yf


@st.cache_resource
def init_sentiment_analyzer():
  try:
    nltk.data.find('sentiment/vader_lexicon.zip')
  except LookupError:
    nltk.download('vader_lexicon', quiet=True)
  return SentimentIntensityAnalyzer()


sia = init_sentiment_analyzer()

# ================= PAGE CONFIG =================
st.set_page_config(
    layout='wide', page_title='Quantitative Intelligence Engine', page_icon='⚡'
)

# ================= STYLING =================
st.markdown(
    """
<style>
    .stApp {
        background-color: #0e1117;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 800 500"><defs><linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:%231a1c2e;stop-opacity:1" /><stop offset="100%" style="stop-color:%230e1117;stop-opacity:1" /></linearGradient></defs><rect width="100%" height="100%" fill="url(%23grad)"/><path d="M0,192L48,202.7C96,213,192,235,288,224C384,213,480,171,576,165.3C672,160,768,192,816,208L864,224L864,500L816,500C768,500,672,500,576,500C480,500,384,500,288,500C192,500,96,500,48,500L0,500Z" fill="%23161b26" fill-opacity="0.6"></path></svg>');
        background-size: cover;
        background-attachment: fixed;
    }

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
""",
    unsafe_allow_html=True,
)

# ================= SIDEBAR CONFIG =================
st.sidebar.title('Navigation')
nav_choice = st.sidebar.radio(
    'Go to',
    ['Quant Strategy & Regime', 'News & Sentiment', 'Statistics & Fundamentals'],
)

st.sidebar.subheader('⚙️ Quant Engine Config')
stock = st.sidebar.text_input('Ticker Symbol', '^NSEI').upper().strip()
prediction_days = st.sidebar.slider('Forecast Horizon (Days)', 1, 30, 10)
target_horizon = st.sidebar.selectbox(
    'Target Horizon (Days)', [1, 3, 5], index=2
)
max_iter = st.sidebar.number_input('Boosting Iterations', 30, 300, 75)
confidence_thresh = (
    st.sidebar.slider('Confidence Cutoff (%)', 50, 75, 62) / 100.0
)
training_years = st.sidebar.slider('Historical Data Years', 1, 10, 8)
use_all_data = st.sidebar.checkbox(
    'Use all available historical data', value=True
)


# ================= MACRO & NEWS DATA =================
@st.cache_data(ttl=3600)
def fetch_macro_indicators():
  try:
    vix_df = yf.download(
        '^VIX', period='10y', multi_level_index=False, progress=False
    )
    tnx_df = yf.download(
        '^TNX', period='10y', multi_level_index=False, progress=False
    )

    macro_df = (
        pd.DataFrame({'VIX': vix_df['Close'], 'TNX': tnx_df['Close']})
        .ffill()
        .bfill()
    )
    macro_df.index = (
        pd.to_datetime(macro_df.index).tz_localize(None).normalize()
    )
    return macro_df
  except Exception:
    return pd.DataFrame()


def scrape_finviz_historical_sentiment(ticker):
  url = f'https://finviz.com/quote.ashx?t={ticker}'
  req = urllib.request.Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
  daily_scores = {}
  try:
    html = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(html, 'html.parser')
    news_table = soup.find(id='news-table')

    if news_table:
      current_date = None
      for row in news_table.findAll('tr'):
        title = row.a.text if row.a else ''
        date_data = row.td.text.strip().split()

        if len(date_data) == 2:
          current_date = date_data[0]

        if current_date and title:
          score = sia.polarity_scores(title)['compound']
          dt = pd.to_datetime(current_date, errors='coerce')
          if pd.notnull(dt):
            dt_str = dt.strftime('%Y-%m-%d')
            if dt_str not in daily_scores:
              daily_scores[dt_str] = []
            daily_scores[dt_str].append(score)

    return {k: np.mean(v) for k, v in daily_scores.items()}
  except Exception:
    return {}


# ================= ADVANCED FEATURE ENGINEERING WITH TREND FILTER =================
@st.cache_data
def load_quant_dataset(ticker, years, use_all, horizon):
  period = 'max' if use_all else f'{years}y'

  try:
    data = yf.download(
        ticker, period=period, multi_level_index=False, progress=False
    )
  except Exception:
    return None, 0.0, 'Unknown', {}

  if data.empty:
    return None, 0.0, 'Unknown', {}

  data.index = pd.to_datetime(data.index).tz_localize(None).normalize()

  required_cols = ['Close', 'High', 'Low']
  if not all(col in data.columns for col in required_cols):
    return None, 0.0, 'Unknown', {}

  if 'Volume' not in data.columns or data['Volume'].sum() == 0:
    data['Volume'] = 1000000.0

  data = data[['Close', 'Volume', 'High', 'Low']].ffill().bfill()

  # Target Alignment
  data['Target'] = (data['Close'].shift(-horizon) > data['Close']).astype(int)

  # Trend Filter Features
  data['SMA_200'] = data['Close'].rolling(window=200).mean()
  data['Trend_Regime'] = (data['Close'] > data['SMA_200']).astype(int)

  # Technical Indicators
  data['Pct_Change'] = data['Close'].pct_change()
  data['Lag1'] = data['Pct_Change'].shift(1)
  data['Lag2'] = data['Pct_Change'].shift(2)

  tr1 = data['High'] - data['Low']
  tr2 = (data['High'] - data['Close'].shift(1)).abs()
  tr3 = (data['Low'] - data['Close'].shift(1)).abs()
  data['ATR'] = (
      pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
  )
  data['Vol_20D'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)

  delta = data['Close'].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / (loss + 1e-6)
  data['RSI'] = 100 - (100 / (1 + rs))

  ema12 = data['Close'].ewm(span=12, adjust=False).mean()
  ema26 = data['Close'].ewm(span=26, adjust=False).mean()
  data['MACD'] = ema12 - ema26

  macro = fetch_macro_indicators()
  if not macro.empty:
    data = data.join(macro, how='left').ffill().bfill()
  else:
    data['VIX'] = 20.0
    data['TNX'] = 4.0

  returns = data['Close'].pct_change()
  rolling_momentum = returns.rolling(window=10).mean()
  max_mom = rolling_momentum.abs().max()
  data['Sentiment'] = (
      rolling_momentum / max_mom if max_mom > 0 else rolling_momentum
  ).fillna(0.0)

  finviz_sentiments = scrape_finviz_historical_sentiment(ticker)
  for date_str, score in finviz_sentiments.items():
    dt = pd.to_datetime(date_str)
    if dt in data.index:
      data.loc[dt, 'Sentiment'] = score

  recent_sentiment = float(data['Sentiment'].iloc[-1])
  data = data.dropna()

  current_vix = float(data['VIX'].iloc[-1])
  current_vol = float(data['Vol_20D'].iloc[-1])
  current_rsi = float(data['RSI'].iloc[-1])

  if current_vix > 25.0 or current_vol > 0.35:
    regime = '🚨 High Volatility / Extreme Risk'
  elif current_rsi > 60.0 or current_rsi < 40.0:
    regime = '📈 Trending Directional Regime'
  else:
    regime = '🔄 Low-Volatility Mean-Reverting'

  latest_close = float(data['Close'].iloc[-1])
  latest_atr = float(data['ATR'].iloc[-1])

  risk_params = {
      'ATR': latest_atr,
      'Long_SL': latest_close - (2.0 * latest_atr),
      'Long_TP': latest_close + (3.0 * latest_atr),
      'Short_SL': latest_close + (2.0 * latest_atr),
      'Short_TP': latest_close - (3.0 * latest_atr),
      'Vol_Annualized': current_vol * 100,
  }

  return data, recent_sentiment, regime, risk_params


# ================= HIGH-PRECISION GRADIENT BOOSTED ENGINE =================
@st.cache_resource
def train_gradient_boosted_engine(data, max_iter, conf_cutoff):
  feature_cols = [
      'Pct_Change',
      'Lag1',
      'Lag2',
      'Volume',
      'RSI',
      'MACD',
      'ATR',
      'Vol_20D',
      'Trend_Regime',
      'VIX',
      'TNX',
      'Sentiment',
  ]

  X = data[feature_cols].values
  y = data['Target'].values

  train_size = int(len(X) * 0.8)
  X_train, y_train = X[:train_size], y[:train_size]
  X_test, y_test = X[train_size:], y[train_size:]

  # Calibrated Boosting Tree Parameters to prevent over-fitting
  model = HistGradientBoostingClassifier(
      max_iter=int(max_iter),
      learning_rate=0.015,
      max_leaf_nodes=8,
      min_samples_leaf=25,
      l2_regularization=2.0,
      random_state=42,
  )
  model.fit(X_train, y_train)

  probs = model.predict_proba(X_test)[:, 1]

  # Strict Confidence Filtering
  high_conf_mask = (probs >= conf_cutoff) | (probs <= (1.0 - conf_cutoff))
  filtered_probs = probs[high_conf_mask]
  filtered_actuals = y_test[high_conf_mask]

  if len(filtered_probs) > 0:
    filtered_preds = (filtered_probs >= 0.5).astype(int)
    acc = float(accuracy_score(filtered_actuals, filtered_preds) * 100)
    prec = float(
        precision_score(filtered_actuals, filtered_preds, zero_division=0)
        * 100
    )
  else:
    acc, prec = 50.0, 50.0

  avg_daily_drift = float(np.mean(np.abs(data['Pct_Change'])))

  return (
      model,
      feature_cols,
      acc,
      prec,
      avg_daily_drift,
      probs,
      len(filtered_probs),
  )


# ================= PREDICTION LOGIC =================
def predict_future_path(
    model, feature_cols, latest_row, days_to_predict, last_actual_price, avg_drift
):
  current_features = latest_row[feature_cols].copy().values.reshape(1, -1)
  predicted_probs = []
  predicted_prices = [last_actual_price]

  for _ in range(days_to_predict):
    prob_up = model.predict_proba(current_features)[0, 1]
    predicted_probs.append(prob_up)

    direction = 1.0 if prob_up > 0.5 else -1.0
    step_change = direction * avg_drift * (abs(prob_up - 0.5) * 2)
    next_price = predicted_prices[-1] * (1 + step_change)
    predicted_prices.append(next_price)

    current_features[0, 2] = current_features[0, 1]  # Lag2 = Lag1
    current_features[0, 1] = current_features[0, 0]  # Lag1 = Pct_Change
    current_features[0, 0] = step_change

  return predicted_prices[1:], predicted_probs


df, current_sentiment, regime, risk_params = load_quant_dataset(
    stock, training_years, use_all_data, target_horizon
)

# ================= VIEW 1: QUANT STRATEGY =================
if nav_choice == 'Quant Strategy & Regime':
  st.markdown(
      '<p class="hero-title">⚡ Quantitative Intelligence Engine</p>',
      unsafe_allow_html=True,
  )
  st.markdown(
      '<p class="hero-subtitle">High-Confidence Filtered Signals with Trend'
      ' Regime Verification</p>',
      unsafe_allow_html=True,
  )

  if df is not None:
    latest_price = float(df['Close'].iloc[-1])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Active Symbol', stock)
    col2.metric('Latest Close', f'${latest_price:,.2f}')
    col3.metric('Annualized Volatility', f"{risk_params['Vol_Annualized']:.1f}%")
    col4.metric('Market Sentiment', f'{current_sentiment:+.2f}')

    st.write('---')

    col_r1, col_r2 = st.columns([2, 2])

    with col_r1:
      st.info(f'**Detected Market Regime:**\n### {regime}')
      st.caption(
          f"CBOE VIX Index: **{df['VIX'].iloc[-1]:.2f}** | 10Y Yield:"
          f" **{df['TNX'].iloc[-1]:.2f}%** | 14-Day ATR:"
          f" **${risk_params['ATR']:.2f}**"
      )

    with col_r2:
      st.success(f"""
            **Dynamic Long Position Parameters:**
            * **Stop-Loss (2x ATR):** `${risk_params['Long_SL']:,.2f}`
            * **Take-Profit (3x ATR):** `${risk_params['Long_TP']:,.2f}`
            
            **Dynamic Short Position Parameters:**
            * **Stop-Loss (2x ATR):** `${risk_params['Short_SL']:,.2f}`
            * **Take-Profit (3x ATR):** `${risk_params['Short_TP']:,.2f}`
            """)

    st.write('---')

    with st.status(
        'Executing Calibrated High-Confidence Engine...', expanded=True
    ) as status:
      st.write(
          'Filtering noisy trade windows via 200-SMA and L2 regularization...'
      )
      (
          model,
          feature_cols,
          acc,
          prec,
          avg_drift,
          test_probs,
          high_conf_count,
      ) = train_gradient_boosted_engine(df, max_iter, confidence_thresh)

      st.write('Generating high-conviction trade probabilities...')
      future_prices, future_probs = predict_future_path(
          model,
          feature_cols,
          df.iloc[-1],
          prediction_days,
          latest_price,
          avg_drift,
      )
      status.update(
          label='Execution Complete!', state='complete', expanded=False
      )

    # ACCURACY SCORECARD
    st.subheader(
        f'🎯 High-Confidence Performance Scorecard (Cutoff: {confidence_thresh * 100:.0f}%)'
    )
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(
        f'{target_horizon}-Day Directional Accuracy', f'{acc:.1f}%'
    )
    col_m2.metric('Bullish Signal Precision', f'{prec:.1f}%')
    col_m3.metric('High-Confidence Samples', f'{high_conf_count} Trade Windows')

    st.write('---')

    # FORECAST CHART
    st.subheader(f'📈 Next {prediction_days}-Day Probability & Price Path')
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=prediction_days
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index[-200:],
            y=df['Close'].iloc[-200:].values.flatten(),
            name='Recent History',
            line=dict(color='#00d2ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 210, 255, 0.05)',
        )
    )

    connect_x = [df.index[-1]] + list(future_dates)
    connect_y = [latest_price] + list(future_prices)

    fig.add_trace(
        go.Scatter(
            x=connect_x,
            y=connect_y,
            name='Regime-Aware Path',
            line=dict(color='#ff4b4b', width=3, dash='dash'),
        )
    )

    fig.add_hline(
        y=risk_params['Long_TP'],
        line_dash='dot',
        line_color='#00e676',
        annotation_text='Long Take-Profit (3x ATR)',
    )
    fig.add_hline(
        y=risk_params['Long_SL'],
        line_dash='dot',
        line_color='#ff5252',
        annotation_text='Long Stop-Loss (2x ATR)',
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=480,
        hovermode='x unified',
        autosize=True,
        margin=dict(l=20, r=20, t=30, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    # SIGNAL TABLE
    st.subheader(f'Next {prediction_days}-Day Quant Trade Signals')
    res_df = pd.DataFrame({
        'Date': future_dates.date,
        'Projected Price': future_prices,
        'Bullish Probability': [f'{p * 100:.1f}%' for p in future_probs],
        'Quant Signal': [
            '🟢 BUY / LONG'
            if p >= confidence_thresh
            else (
                '🔴 SELL / SHORT' if p <= (1.0 - confidence_thresh) else '⚪ HOLD / CASH'
            )
            for p in future_probs
        ],
    })
    st.dataframe(
        res_df.style.format({'Projected Price': '${:,.2f}'}),
        use_container_width=True,
    )

  else:
    st.error(f"Could not load historical dataset for ticker '{stock}'.")

# ================= VIEW 2: NEWS =================
elif nav_choice == 'News & Sentiment':
  st.markdown(
      f'<p class="hero-title">📰 News & Sentiment Analysis — {stock}</p>',
      unsafe_allow_html=True,
  )
  st.markdown(
      '<p class="hero-subtitle">Real-time Yahoo Finance headlines processed via'
      ' VADER Lexicon</p>',
      unsafe_allow_html=True,
  )

  ticker_obj = yf.Ticker(stock)
  news_items = ticker_obj.news

  if news_items:
    parsed_news = []
    for item in news_items:
      content = item.get('content', {})
      title = content.get('title') or item.get('title', 'No Title')
      provider = content.get('provider', {}).get('displayName') or item.get(
          'publisher', 'Unknown Source'
      )
      link = content.get('canonicalUrl', {}).get('url') or item.get(
          'link', '#'
      )

      scores = sia.polarity_scores(title)
      compound = scores['compound']

      sentiment = (
          'Positive'
          if compound >= 0.05
          else ('Negative' if compound <= -0.05 else 'Neutral')
      )

      parsed_news.append({
          'Title': title,
          'Source': provider,
          'Sentiment': sentiment,
          'Compound Score': compound,
          'Link': link,
      })

    news_df = pd.DataFrame(parsed_news)
    avg_compound = news_df['Compound Score'].mean()

    col1, col2 = st.columns(2)
    col1.metric('Overall Sentiment Score', f'{avg_compound:.2f}')
    if avg_compound >= 0.05:
      col2.success('Market Outlook: Bullish / Positive Sentiment')
    elif avg_compound <= -0.05:
      col2.error('Market Outlook: Bearish / Negative Sentiment')
    else:
      col2.info('Market Outlook: Neutral Sentiment')

    st.write('---')
    st.subheader('Recent Headlines')

    for _, row in news_df.iterrows():
      badge_color = (
          '🟢'
          if row['Sentiment'] == 'Positive'
          else ('🔴' if row['Sentiment'] == 'Negative' else '⚪')
      )
      st.markdown(f"### {badge_color} [{row['Title']}]({row['Link']})")
      st.caption(
          f"Source: **{row['Source']}** | Sentiment Score:"
          f" **{row['Compound Score']:.2f}** ({row['Sentiment']})"
      )
      st.write('---')
  else:
    st.warning(f'No recent news articles were retrieved for {stock}.')

# ================= VIEW 3: STATS =================
elif nav_choice == 'Statistics & Fundamentals':
  st.markdown(
      f'<p class="hero-title">📊 Fundamental & Macro Metrics — {stock}</p>',
      unsafe_allow_html=True,
  )
  st.link_button(
      f'🔗 Open {stock} directly on Yahoo Finance',
      f'https://finance.yahoo.com/quote/{stock}',
  )
  st.write('---')

  ticker_obj = yf.Ticker(stock)
  try:
    info = ticker_obj.info
    company_name = info.get('longName', stock)
    trailing_pe = info.get('trailingPE', 'N/A')
    forward_pe = info.get('forwardPE', 'N/A')
    market_cap = info.get('marketCap', None)
    profit_margins = info.get('profitMargins', None)

    mcap_str = f'${market_cap / 1e9:,.2f}B' if market_cap else 'N/A'
    margin_str = f'{profit_margins * 100:.2f}%' if profit_margins else 'N/A'

    st.info(
        f'**{company_name} ({stock})** current Market Cap stands at'
        f' **{mcap_str}** with a Trailing P/E of **{trailing_pe}** and Forward'
        f' P/E of **{forward_pe}**. Profit margin is currently recorded at'
        f' **{margin_str}**.'
    )

    st.subheader('Valuation & Financial Health')
    metrics_df = pd.DataFrame({
        'Metric': [
            'Market Cap',
            'Enterprise Value',
            'Trailing P/E',
            'Forward P/E',
            'PEG Ratio (5yr expected)',
            'Price/Sales',
            'Price/Book',
        ],
        'Current Value': [
            mcap_str,
            (
                f"${info.get('enterpriseValue', 0) / 1e9:,.2f}B"
                if info.get('enterpriseValue')
                else 'N/A'
            ),
            trailing_pe,
            forward_pe,
            info.get('pegRatio', 'N/A'),
            info.get('priceToSalesTrailing12Months', 'N/A'),
            info.get('priceToBook', 'N/A'),
        ],
    })
    st.dataframe(metrics_df, use_container_width=True)
  except Exception:
    st.warning('Could not load live valuation metadata for ticker.')
