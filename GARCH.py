import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

class GARCH:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = yf.Ticker(ticker)
        self.start_date = start_date
        self.end_date = end_date
        self.returns = None
        
    def get_returns(self):
        data = self.ticker.history(start=self.start_date, end=self.end_date)
        self.returns = data['Close'].pct_change().dropna()
        return self.returns
    
    def get_actual_volatility(self, returns):
        return np.sqrt(returns.rolling(window=21).var())