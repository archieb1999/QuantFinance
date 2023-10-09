import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import datetime


def download_data(stock, start_date, end_date):
    data = {}
    ticker = yf.download(stock, start_date, end_date)
    data[stock] = ticker['Adj Close']
    return pd.DataFrame(data)


def calculate_var(position, c, mu, sigma, n):
    var = position * (-mu*n - sigma * np.sqrt(n) * norm.ppf(1 - c))
    return var

if __name__ == '__main__':
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2022,3,1)

    stock_data = download_data('C', start, end)

    stock_data['returns'] = np.log(stock_data['C'] / stock_data['C'].shift(1))

    stock_data = stock_data[1:]

    S = 1e6

    c = 0.99

    mu = np.mean(stock_data['returns'])
    sigma = np.std(stock_data['returns'])
    print(mu, sigma, norm.ppf(1-c))
    print('value at risk is $%.2f' % calculate_var(S, c, mu, sigma, 1))