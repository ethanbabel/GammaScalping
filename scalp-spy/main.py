from AlgorithmImports import *
from datetime import timedelta
from GARCH import GARCH
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import monte_carlo
import cost_projection

class Scalpspy(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2016, 1, 1)  # Set Start Date
        self.set_end_date(2016, 12, 31)  # Set End Date
        self.set_cash(100000)  # Set Strategy Cash
        self.spy = self.AddEquity("SPY", Resolution.DAILY).Symbol
        self.set_benchmark(self.spy)  # Set Benchmark
        option = self.AddOption("SPY", Resolution.HOUR)
        option.set_filter(3, 10, timedelta(2), timedelta(21))
        self.set_risk_free_interest_rate_model(InterestRateProvider())
        self.myOptions = {}
        self.myOptionsDeltas = {}
        self.margin_buffer = 1


        # Load GARCH model, past data
        self.garch = GARCH("SPY")
        self.rolling_window_size = 365
        self.rolling_window = RollingWindow[TradeBar](self.rolling_window_size)
        history_trade_bar = self.history[TradeBar](self.spy, self.rolling_window_size, Resolution.DAILY)
        for bar in history_trade_bar:
            self.rolling_window.Add(bar)
        
        

    def on_data(self, data: Slice):
        self.rebalance_and_check_margin(data)

        # # Liquidate Options if needed
        # if self.Portfolio.Cash < self.keep_cash:
        #     worst_current_option = self.get_worst_current_option()
        #     if worst_current_option:
        #         self.Liquidate(worst_current_option)
        #         self.Debug(f"Liquidated {worst_current_option.Symbol} because cash is low")
        #         del self.myOptions[worst_current_option.Symbol]
        #         del self.myOptionsDeltas[worst_current_option.Symbol]

        # Liquidate options that are too close to expiration
        temp = set()
        for option in self.myOptions.values():
            if self.Time + timedelta(2) > option.Expiry:
                self.Liquidate(option, "Too close to expiration")
                self.Debug(f"Liquidated {option.Symbol} because it is too close to expiration")
                temp.add(option.Symbol)
                
        for option in temp:
            del self.myOptions[option]
            del self.myOptionsDeltas[option]

        # At Market close check for new options
        if self.Time.hour == 16:
            self.daily(data)
            
        # Set Portfolio to Delta Neutral
        self.delta_balance_portfolio(data)

    def daily(self, data: Slice):
        # Update GARCH model
        self.rolling_window.Add(data["SPY"])
        close_prices = pd.DataFrame([x.Close for x in self.rolling_window], columns=["Close"])
        close_prices = close_prices.iloc[::-1]
        rolling_vol = 100 * close_prices.pct_change().dropna()
        self.garch.train(rolling_vol, 2, 2)

        if not data.OptionChains:
            self.Debug("No options chains data")
        
        if self.is_margin_low():
            self.rebalance_and_check_margin(data)
            return
        
        # Parse option chains to find best option to buy
        for chain in data.OptionChains:

            # Filter otm calls with proper expiry
            otm_calls = [x for x in chain.Value if x.Right == OptionRight.Call and x.UnderlyingLastPrice < x.Strike]
            sorted_contracts = sorted(otm_calls, key=lambda x: x.Greeks.Gamma / abs(x.Greeks.Theta) if x.Greeks.Theta != 0 else 0, reverse=True) # Sort by gamma/theta
            filtered_contracts = [x for x in sorted_contracts if x.Expiry > self.Time + timedelta(3) and x.Expiry < self.Time + timedelta(21)]
            if len(filtered_contracts) == 0:
                self.Debug("No contracts found")
                self.Debug("otm_calls: " + str(len(otm_calls)))
                self.Debug("sorted_contracts: " + str(len(sorted_contracts)))
                self.Debug("filtered_contracts: " + str(len(filtered_contracts)))
                continue

            # Find best contract to buy
            profits = {}
            for contract in filtered_contracts:
                IV = contract.ImpliedVolatility
                predicted_profits = monte_carlo.monte_carlo_gamma_scalping(contract.UnderlyingLastPrice, IV, contract.Greeks.Gamma, contract.Greeks.Theta, 1, 21, 10000)
                predicted_profit = np.mean(predicted_profits)
                profits[contract] = predicted_profit
            best_contract = max(profits, key=profits.get)

            # Calculate cost of gamma scalping
            current_datetime = datetime(self.Time.year, self.Time.month, self.Time.day, tzinfo=pytz.timezone('America/New_York')).date()
            time_horizon = (best_contract.Expiry - self.Time).days - 2
            cash_per_contract = cost_projection.gamma_scalping_cost(self.garch, current_datetime, time_horizon, best_contract.UnderlyingLastPrice, best_contract.LastPrice, best_contract.Greeks.Gamma, best_contract.Greeks.Delta)
            max_possible_num_contracts =  self.get_buying_power() // cash_per_contract

            if max_possible_num_contracts < 1 and profits[best_contract] > 0:
                # If we can't afford any contracts, check to see if worst currently held contract is worse than the best contract
                worst_current_option = self.get_worst_current_option()
                if worst_current_option is None:
                    continue
                IV = worst_current_option.ImpliedVolatility
                worst_current_predicted_profits = monte_carlo.monte_carlo_gamma_scalping(worst_current_option.UnderlyingLastPrice, IV, worst_current_option.Greeks.Gamma, worst_current_option.Greeks.Theta, 1, 21, 10000)
                worst_current_predicted_profit = np.mean(worst_current_predicted_profits)
                
                if worst_current_predicted_profit < profits[best_contract]:
                    self.Liquidate(worst_current_option)
                    self.Debug(f"Liquidated {worst_current_option.Symbol} to buy {best_contract.Symbol}")
                    del self.myOptions[worst_current_option.Symbol]
                    del self.myOptionsDeltas[worst_current_option.Symbol]

                     # Re-Calculate cost of gamma scalping
                    cash_per_contract = cost_projection.gamma_scalping_cost(self.garch, current_datetime, time_horizon, best_contract.UnderlyingLastPrice, best_contract.LastPrice, best_contract.Greeks.Gamma, best_contract.Greeks.Delta)
                    max_possible_num_contracts =  self.get_buying_power() // cash_per_contract
                    if max_possible_num_contracts <= 0:
                        continue
                    quantity = max_possible_num_contracts
                    self.Buy(best_contract.Symbol, quantity)
                    self.myOptions[best_contract.Symbol] = best_contract
                    self.myOptionsDeltas[best_contract.Symbol] = best_contract.Greeks.Delta
                    self.Debug(f"Bought {quantity} of {best_contract.Symbol} with IV: {IV}  and Predicted Profit: {profits[best_contract]}")

            elif max_possible_num_contracts >= 1 and profits[best_contract] > 0:
                self.myOptions[best_contract.Symbol] = best_contract
                self.myOptionsDeltas[best_contract.Symbol] = best_contract.Greeks.Delta
                quantity = max_possible_num_contracts
                self.Buy(best_contract.Symbol, quantity)
                self.Debug(f"Bought {quantity} of {best_contract.Symbol} with IV: {IV}  and Predicted Profit: {profits[best_contract]}")
            else:
                self.Debug(f"Did not buy best contract {best_contract.Symbol} because predicted profit was negative")

    def delta_balance_portfolio(self, data: Slice):
        # Set Portfolio to Delta Neutral

        if len(self.myOptions) != 0:
            self.get_options_deltas(data)
            for symbol in self.myOptions.keys():
                self.Debug(f"Option: {symbol} Delta: {self.myOptionsDeltas[symbol]}, Quantity: {self.Portfolio[symbol].Quantity}")
            options_delta = sum([pair[1] * 100 * self.Portfolio[pair[0]].Quantity for pair in self.myOptionsDeltas.items()])
            underlying_delta = self.Portfolio["SPY"].Quantity
            portfolio_delta = options_delta + underlying_delta
            self.Debug(f"Options Delta: {options_delta}")
            self.Debug(f"Underlying Delta: {underlying_delta}")
            if abs(portfolio_delta) > 1:
                self.MarketOrder("SPY", round(-portfolio_delta))
        else:
            self.Liquidate()
    
    def get_options_deltas(self, data):
        for chain in data.OptionChains:
            for contract in chain.Value:
                if contract.Symbol in self.myOptionsDeltas:
                    self.myOptionsDeltas[contract.Symbol] = contract.Greeks.Delta
            
    def get_worst_current_option(self):
        curr_profits = {}
        for contract in self.myOptions.values():
            IV = contract.ImpliedVolatility
            curr_predicted_profits = monte_carlo.monte_carlo_gamma_scalping(contract.UnderlyingLastPrice, IV, contract.Greeks.Gamma, contract.Greeks.Theta, 1, 21, 10000)
            curr_predicted_profit = np.mean(curr_predicted_profits)
            curr_profits[contract] = curr_predicted_profit
        worst_current_contract = None
        if curr_profits:
            worst_current_contract = min(curr_profits, key=curr_profits.get)
        return worst_current_contract

    def rebalance_and_check_margin(self, data: Slice):
        if (self.is_margin_low()):
            self.Debug("Margin too low, liquidating options")
            while(self.is_margin_low()):
                worst_option = self.get_worst_current_option()
                if worst_option is None:
                    break
                self.MarketOrder(worst_option.Symbol, -1)
                del self.myOptions[worst_option.Symbol]
                del self.myOptionsDeltas[worst_option.Symbol]
                worst_option = self.get_worst_current_option()
            self.delta_balance_portfolio(data)
    
    def is_margin_low(self):
        potential_max_options_delta = sum([0.5 * 100 * self.Portfolio[option.Symbol].Quantity for option in self.myOptions.values()])
        potential_max_short_position  = potential_max_options_delta * self.Securities["SPY"].Price 
        rtn = (self.get_buying_power()) * self.margin_buffer < potential_max_short_position
        if rtn:
            self.Debug(f"Margin too high! Margin: {self.get_buying_power()}, Potential Short Position: {potential_max_short_position}")
        return rtn

    def get_buying_power(self):
        spy = self.Securities["SPY"]
        return self.Portfolio.GetBuyingPower(spy.Symbol, OrderDirection.SELL) * self.margin_buffer