import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from GARCH import GARCH

class Ticker:
    def __init__(self, ticker, p, q):
        self.ticker = yf.Ticker(ticker)
        self.data = None
        self.garch = GARCH(ticker)
        self.p = p
        self.q = q

    def get_data(self, period="1y"):
        import yfinance as yf
        self.data = self.ticker.history(period=period).Close
        self.data = 100 * self.data.pct_change().dropna()

    def get_1w_prediction(self):
        self.garch.train(self.data, self.p, self.q)
        return self.garch.predict(7)

if __name__ == "__main__":
    ticker = Ticker("SPY", 1, 1)
    ticker.get_data()
    print(ticker.get_1w_prediction())