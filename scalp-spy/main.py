from AlgorithmImports import *
from datetime import timedelta
from GARCH import GARCH
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

class Scalpspy(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)  # Set Start Date
        self.set_end_date(2016, 12, 31)  # Set End Date
        self.set_cash(100000)  # Set Strategy Cash
        spy = self.AddEquity("SPY", Resolution.DAILY).Symbol
        option = self.AddOption("SPY", Resolution.HOUR)
        option.set_filter(2, 2, timedelta(10), timedelta(40))
        self.myOptions = {}
        self.myOptionsDeltas = {}

        # Load GARCH model, past data
        self.garch = GARCH("SPY")
        self.rolling_window_size = 365
        self.rolling_window = RollingWindow[TradeBar](self.rolling_window_size)
        history_trade_bar = self.history[TradeBar](spy, self.rolling_window_size, Resolution.DAILY)
        for bar in history_trade_bar:
            self.rolling_window.Add(bar)
        
        

    def on_data(self, data: Slice):
        # At Market close check for new options
        if self.Time.hour == 16:
            self.daily(data)

        # Liquidate options that are too close to expiration
        for option in self.myOptions.values():
            if self.Time + timedelta(2) > option.Expiry:
                self.Liquidate(option, "Too close to expiration")

        # Set Portfolio to Delta Neutral
        if len(self.myOptions) != 0:
            self.get_options_deltas(data)
            for symbol in self.myOptions.keys():
                self.Debug(f"Option: {symbol} Delta: {self.myOptions[symbol]}, Quantity: {self.Portfolio[symbol].Quantity}")
            options_delta = sum([pair[1] * 100 * self.Portfolio[pair[0]].Quantity for pair in self.myOptionsDeltas.items()])
            underlying_delta = self.Portfolio["SPY"].Quantity
            portfolio_delta = options_delta + underlying_delta
            self.Debug(f"Options Delta: {options_delta}")
            self.Debug(f"Underlying Delta: {underlying_delta}")
            if abs(portfolio_delta) > 0.1:
                self.MarketOrder("SPY", -portfolio_delta)

    def daily(self, data: Slice):
        # Update GARCH model
        self.rolling_window.Add(data["SPY"])
        close_prices = pd.DataFrame([x.Close for x in self.rolling_window], columns=["Close"])
        close_prices = close_prices.iloc[::-1]
        rolling_vol = 100 * close_prices.pct_change().dropna()
        self.garch.train(rolling_vol, 2, 2)
        current_datetime = datetime(self.Time.year, self.Time.month, self.Time.day, tzinfo=pytz.timezone('America/New_York')).date()
        GARCH_pred = self.garch.predict(time_horizon=21, startdate=current_datetime) # Predict next 3 weeks volatility
        
        # If predicted volatility is high, buy options
        for chain in data.OptionChains:
            otm_calls = [x for x in chain.Value if x.Right == OptionRight.Call and x.UnderlyingLastPrice < x.Strike]
            sorted_contracts = sorted(otm_calls, key=lambda x: x.Greeks.Gamma / abs(x.Greeks.Theta) if x.Greeks.Theta != 0 else 0, reverse=True) # Sort by gamma/theta
            filtered_contracts = [x for x in sorted_contracts if x.Expiry > self.Time + timedelta(7) and x.Expiry < self.Time + timedelta(21)]
            if len(filtered_contracts) == 0:
                self.Debug("No contracts found")
                continue
            for contract in filtered_contracts:
                IV = contract.ImpliedVolatility
                GARCH_IV = GARCH_pred[contract.Expiry.date()]
                if IV < GARCH_IV:
                    self.myOptions[contract.Symbol] = contract
                    self.myOptionsDeltas[contract.Symbol] = contract.Greeks.Delta
                    self.Buy(contract.Symbol, 5)
                    self.Debug(f"Bought {contract.Symbol} with IV: {IV} and Predicted IV: {GARCH_IV}")
                    break
                else:
                    self.Debug(f"IV too high for {contract.Symbol} with IV: {IV} and Predicted IV: {GARCH_IV}")

    def get_options_deltas(self, data):
        for chain in data.OptionChains:
            for contract in chain.Value:
                if contract.Symbol in self.myOptionsDeltas:
                    self.myOptionsDeltas[contract.Symbol] = contract.Greeks.Delta
            

         

    
