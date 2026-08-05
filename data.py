import yfinance as yf

def get_stock_data(stock):
    data = yf.download(stock, start="2018-01-01")
    return data[['Close']]