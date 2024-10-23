from AlgorithmImports import *
from datetime import timedelta
from QuantConnect.Data.Custom.CBOE import * 

class Scalpspy(QCAlgorithm):

    def initialize(self):
        # Locally Lean installs free sample data, to download more data please visit https://www.quantconnect.com/docs/v2/lean-cli/datasets/downloading-data
        self.set_start_date(2013, 10, 7)  # Set Start Date
        self.set_end_date(2013, 10, 11)  # Set End Date
        self.set_cash(100000)  # Set Strategy Cash
        option = self.AddOption("SPY")
        option.set_filter(2, 2, timedelta(10), timedelta(40))

    def on_data(self, data: Slice):
        options_invested = [x.Key for x in self.Portfolio if x.Value.Invested and x.Value.Type == SecurityType.Option]

        # Liquidate options that are too close to expiration
        for option in options_invested:
            if self.Time + timedelta(10) > option.ID.Date:
                self.Liquidate(option, "Too close to expiration")
        
        # Set Portfolio to Delta Neutral
        delta = sum(self.Securities[option].Greeks.Delta * self.Portfolio[option].Quantity for option in options_invested)
        self.Debug(f"Current Delta Position: {delta}")
                
         


