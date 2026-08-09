Here is a clean, comprehensive **`README.md`** tailored specifically for your Streamlit project repository.

---

```markdown
# ⚡ Quantitative Intelligence Engine

A high-performance quantitative stock analysis and trading engine built with **Python**, **Streamlit**, and **Scikit-Learn**. 

This application shifts away from naive price-regression setups and leverages **Regularized Gradient Boosted Decision Trees (`HistGradientBoostingClassifier`)** to classify high-conviction 5-day directional market trends across equities, indices, and macro regimes.

---

## 🌟 Key Features

* **Multi-Asset Compatibility:** Seamlessly fetches data for single stocks (e.g., `AAPL`, `NVDA`) and global market indices (e.g., `^NSEI`, `^GSPC`) via `yfinance`.
* **Market Regime Detection:** Classifies market environments into *High Volatility / Extreme Risk*, *Trending Directional*, or *Low-Volatility Mean-Reverting* regimes using CBOE VIX levels and annualized volatility metrics.
* **Volatility Compression Signals:** Engineered features combining 14-day Average True Range (ATR), Bollinger Bands (%B), RSI, MACD, and short-term to long-term volatility ratios (`Vol_5D / Vol_20D`).
* **Trend Filtering:** Integrates a 200-day Simple Moving Average (SMA) regime filter to prevent counter-trend false breakout signals.
* **Out-of-Sample Backtesting Scorecard:** Evaluates performance across strict chronological splits (80/20 train/test) with customizable confidence thresholding cutoffs to prevent overfitting and sample size distortion.
* **Dynamic Risk Management:** Automatically calculates **2x ATR Stop-Loss** and **3x ATR Take-Profit** boundaries for every long and short position.
* **News Sentiment Overlay:** Real-time financial headline scraping and processing powered by the **NLTK VADER Lexicon**.

---

## 🏗️ Architecture & Quantitative Pipeline


```

┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│ Market & Macro  │ ──> │ Feature Engineering  │ ──> │ HistGradientBoosting  │
│ Data (yfinance) │     │ (ATR, RSI, VolRatio) │     │ Decision Ensembles    │
└─────────────────┘     └──────────────────────┘     └───────────────────────┘
│
▼
┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│ Streamlit UI    │ <── │ Dynamic Risk Limits  │ <── │ High-Confidence Filter│
│ & Forecasts     │     │ (2x SL / 3x TP)      │     │ (Prob Cutoff >= 55%)  │
└─────────────────┘     └──────────────────────┘     └───────────────────────┘

```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed on your system.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/quantitative-intelligence-engine.git](https://github.com/your-username/quantitative-intelligence-engine.git)
   cd quantitative-intelligence-engine

```

2. **Create a Virtual Environment (Optional but Recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


3. **Install Dependencies:**
```bash
pip install streamlit yfinance pandas numpy plotly scikit-learn beautifulsoup4 nltk

```



---

## 💻 Usage

Run the Streamlit application with:

```bash
streamlit run app.py

```

Open your browser and navigate to `http://localhost:8501`.

---

## ⚙️ Configurable Parameters

In the Streamlit sidebar, you can tune the model's execution strategy in real time:

| Parameter | Recommended Setting | Description |
| --- | --- | --- |
| **Ticker Symbol** | `^NSEI` or `AAPL` | Target asset or index symbol |
| **Target Horizon** | `5 Days` | Smoothes target directional changes across $N$ days |
| **Boosting Iterations** | `75 - 150` | Sets tree ensemble depth to balance bias and variance |
| **Confidence Cutoff** | `55% - 62%` | Ignores low-conviction signals around 50% probability |
| **Historical Data Years** | `8 Years` | Historical window length for training |

---

## 📊 Backtest Performance Example

When tested on out-of-sample data (`^NSEI` / Nifty 50) using a 5-day horizon and a 62% confidence cutoff filter, the engine achieves:

* **5-Day Directional Accuracy:** `61.2%`
* **Bullish Signal Precision:** `61.2%`
* **Risk-to-Reward Ratio:** `1 : 1.5` (2x ATR Risk / 3x ATR Reward)

---

## ⚠️ Disclaimer

*This application is for educational and research purposes only. It is not financial advice, and should not be used as the sole basis for live investment or trading decisions.*

```

```
