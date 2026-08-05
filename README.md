# Accelerated Stock Predictor

A Streamlit stock price prediction app using an LSTM neural network and Yahoo Finance data.

## Overview

This project includes:
- `app.py` — main Streamlit application with interactive sidebar controls and Plotly visualizations.
- `data.py` — helper for loading historical stock data from Yahoo Finance.
- `model.py` — helper for training and predicting using an LSTM model.

## Features

- Enter any valid ticker symbol (default `AAPL`).
- Configure prediction days, training epochs, and batch size.
- View historical closing price trends and future forecast curves.
- Access live valuation metrics and Yahoo Finance analysis links.

## Requirements

- Python 3.10+
- streamlit
- numpy
- pandas
- yfinance
- plotly
- scikit-learn
- tensorflow

## Setup

1. Clone the repository.
2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the environment:

- Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

- Windows Command Prompt:

```bat
venv\Scripts\activate.bat
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the app:

```bash
streamlit run app.py
```

## Notes


- If the app fails to load ticker data, verify that the ticker symbol is valid.
- This app uses Yahoo Finance data via `yfinance`, so an internet connection is required.
