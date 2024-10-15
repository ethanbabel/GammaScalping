import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from arch import arch_model
import numpy as np

class GARCH:
    def __init__(self, ticker):
        self.ticker = ticker
        self.model = None
        self.model_fit = None
    
    def train(self, training_data, p, q):
        self.model = arch_model(training_data, p=p, q=q)
        self.model_fit = self.model.fit(disp='off')

    def predict(self):
        pred = self.model_fit.forecast(horizon=7)
        todays_date = datetime.today().date()
        future_dates = [todays_date + timedelta(days=i) for i in range(1,8)]
        pred = pd.Series(np.sqrt(pred.variance.values[-1,:]), index=future_dates)
        return pred

if __name__ == "__main__":
    spy = yf.Ticker("SPY")
    data = 100 * spy.history(period="1y").Close.pct_change().dropna()
    garch = GARCH(spy)
    garch.train(data, 2, 2)
    pred = garch.predict()
    print(pred)
    
    