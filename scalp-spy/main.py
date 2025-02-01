from AlgorithmImports import *
from datetime import timedelta
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import monte_carlo


class Scalpspy(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2017, 1, 1)  # Set Start Date
        self.set_end_date(2017, 6, 30)  # Set End Date
        self.set_cash(100000)  # Set Strategy Cash
        # self.SetBrokerageModel(BrokerageName.ALPACA, AccountType.MARGIN)
        # self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        spy = self.AddEquity("SPY", Resolution.HOUR)
        self.spy = spy.Symbol
        spy.SetShortableProvider(LocalDiskShortableProvider("Alpaca"))
        # spy.SetShortableProvider(InteractiveBrokersShortableProvider())
        # spy.setBuyingPowerModel(BuyingPowerModel.)
        self.set_benchmark(self.spy)  # Set Benchmark
        option = self.AddOption("SPY", Resolution.HOUR)
        self.liquidation_buffer = 7
        option.set_filter(3, 10, timedelta(self.liquidation_buffer), timedelta(90))
        self.set_risk_free_interest_rate_model(InterestRateProvider())
        self.myOptions = {}
        self.myOptionsDeltas = {}
        self.margin_buffer = 0.1
        self.per_trade_allocation = 0.1
        
        

    def on_data(self, data: Slice):

        # Liquidate options that are too close to expiration
        temp = set()
        for option in self.myOptions.values():
            if self.Time + timedelta(self.liquidation_buffer) > option.Expiry:
                self.Liquidate(option, "Too close to expiration")
                # self.Debug(f"Liquidated {option.Symbol} because it is too close to expiration")
                temp.add(option.Symbol)
                
        for option in temp:
            del self.myOptions[option]
            del self.myOptionsDeltas[option]

        # Rebalance if margin is too low
        self.rebalance_and_check_margin(data)

        # At Market close check for new options
        if self.Time.hour == 12:
            self.daily(data)
            
        # Set Portfolio to Delta Neutral
        # self.delta_balance_portfolio(data)
        self.rebalance_and_check_margin(data)

    def on_margin_call_warning(self):
        self.Debug("Margin Call Warning")
        self.rebalance_and_check_margin(self.CurrentSlice)

    def daily(self, data: Slice):
        # Get Risk Free Rate
        interest_rate = self.risk_free_interest_rate_model.get_interest_rate(self.Time)

        # Get today's date
        current_datetime_naive = datetime(self.Time.year, self.Time.month, self.Time.day, self.Time.hour, self.Time.minute, self.Time.second)

        # Get Option Chains
        if not data.OptionChains:
            # self.Debug("No options chains data")
            pass
        
        if self.is_margin_low():
            self.rebalance_and_check_margin(data)
            return
        
        # Parse option chains to find best option to buy
        for chain in data.OptionChains:

            # Filter otm calls with proper expiry
            otm_calls = [x for x in chain.Value if x.Right == OptionRight.Call and x.UnderlyingLastPrice < x.Strike]
            sorted_contracts = sorted(otm_calls, key=lambda x: x.Greeks.Gamma / abs(x.Greeks.Theta) if x.Greeks.Theta != 0 else 0, reverse=True) # Sort by gamma/theta
            filtered_contracts = [x for x in sorted_contracts if x.Expiry > self.Time + timedelta(10) and x.Expiry < self.Time + timedelta(90)]
            if len(filtered_contracts) == 0:
                # self.Debug("No contracts found")
                # self.Debug("otm_calls: " + str(len(otm_calls)))
                # self.Debug("sorted_contracts: " + str(len(sorted_contracts)))
                # self.Debug("filtered_contracts: " + str(len(filtered_contracts)))
                continue

            # Find best contract to buy
            profits = {}
            for contract in filtered_contracts:
                IV = contract.ImpliedVolatility
                predicted_profits = monte_carlo.monte_carlo_gamma_scalping(IV, contract.Greeks.Delta, contract.Expiry - timedelta(self.liquidation_buffer), contract.Expiry, interest_rate, contract.LastPrice, contract.UnderlyingLastPrice, contract.Strike, current_datetime_naive)
                predicted_profit = np.mean(predicted_profits)
                profits[contract] = predicted_profit
            best_contract = max(profits, key=profits.get)

            # Calculate cost of gamma scalping
            current_datetime = datetime(self.Time.year, self.Time.month, self.Time.day, tzinfo=pytz.timezone('America/New_York')).date()
            time_horizon = (best_contract.Expiry - self.Time).days - 2
            # cash_per_contract = cost_projection.gamma_scalping_cost(self.garch, current_datetime, time_horizon, best_contract.UnderlyingLastPrice, best_contract.LastPrice, best_contract.Greeks.Gamma, best_contract.Greeks.Delta)
            # max_possible_num_contracts =  self.get_buying_power() // cash_per_contract
            max_possible_num_contracts = round(self.get_shortable_quantity() // 100 * self.per_trade_allocation)

            if max_possible_num_contracts < 1 and profits[best_contract] > 0:
                # If we can't afford any contracts, check to see if worst currently held contract is worse than the best contract
                worst_current_option = self.get_worst_current_option(interest_rate, current_datetime_naive)
                if worst_current_option is None:
                    continue
                IV = worst_current_option.ImpliedVolatility
                worst_current_predicted_profits = monte_carlo.monte_carlo_gamma_scalping(IV, contract.Greeks.Delta, contract.Expiry - timedelta(self.liquidation_buffer), contract.Expiry, interest_rate, contract.LastPrice, contract.UnderlyingLastPrice, contract.Strike, current_datetime_naive)
                worst_current_predicted_profit = np.mean(worst_current_predicted_profits)
                
                if worst_current_predicted_profit < profits[best_contract]:
                    self.Liquidate(worst_current_option)
                    # self.Debug(f"Liquidated {worst_current_option.Symbol} to buy {best_contract.Symbol}")
                    del self.myOptions[worst_current_option.Symbol]
                    del self.myOptionsDeltas[worst_current_option.Symbol]
                    # self.delta_balance_portfolio(data)
                    self.rebalance_and_check_margin(data)

                    # Re-Calculate max_possible_num_contracts
                    # cash_per_contract = cost_projection.gamma_scalping_cost(self.garch, current_datetime, time_horizon, best_contract.UnderlyingLastPrice, best_contract.LastPrice, best_contract.Greeks.Gamma, best_contract.Greeks.Delta)
                    # max_possible_num_contracts =  self.get_buying_power() // cash_per_contract
                    max_possible_num_contracts = round(self.get_shortable_quantity() // 100 * self.per_trade_allocation)
                    if max_possible_num_contracts <= 0:
                        continue
                    quantity = max_possible_num_contracts
                    self.Buy(best_contract.Symbol, quantity)
                    self.myOptions[best_contract.Symbol] = best_contract
                    self.myOptionsDeltas[best_contract.Symbol] = best_contract.Greeks.Delta
                    # self.Debug(f"Bought {quantity} of {best_contract.Symbol} with IV: {IV}  and Predicted Profit: {profits[best_contract]}")

            if max_possible_num_contracts >= 1 and profits[best_contract] > 0:
                self.myOptions[best_contract.Symbol] = best_contract
                self.myOptionsDeltas[best_contract.Symbol] = best_contract.Greeks.Delta
                quantity = max_possible_num_contracts
                self.Buy(best_contract.Symbol, quantity)
                # self.Debug(f"Bought {quantity} of {best_contract.Symbol} with IV: {IV}  and Predicted Profit: {profits[best_contract]}")
            else:
                # self.Debug(f"Did not buy best contract {best_contract.Symbol} because predicted profit was negative")
                pass

    def delta_balance_portfolio(self, data: Slice):
        # Set Portfolio to Delta Neutral

        if len(self.myOptions) != 0:
            self.get_options_deltas(data)
            for symbol in self.myOptions.keys():
                # self.Debug(f"Option: {symbol} Delta: {self.myOptionsDeltas[symbol]}, Quantity: {self.Portfolio[symbol].Quantity}")
                break
            options_delta = sum([pair[1] * 100 * self.Portfolio[pair[0]].Quantity for pair in self.myOptionsDeltas.items()])
            underlying_delta = self.Portfolio["SPY"].Quantity
            portfolio_delta = options_delta + underlying_delta
            # self.Debug(f"Options Delta: {options_delta}")
            # self.Debug(f"Underlying Delta: {underlying_delta}")
            if abs(portfolio_delta) > 1:
                self.MarketOrder("SPY", round(-portfolio_delta))
        else:
            self.Liquidate()
    
    def get_options_deltas(self, data):
        for chain in data.OptionChains:
            for contract in chain.Value:
                if contract.Symbol in self.myOptionsDeltas:
                    self.myOptionsDeltas[contract.Symbol] = contract.Greeks.Delta
            
    def get_worst_current_option(self, interest_rate, current_datetime_naive):
        curr_profits = {}
        for contract in self.myOptions.values():
            IV = contract.ImpliedVolatility
            curr_predicted_profits = monte_carlo.monte_carlo_gamma_scalping(IV, contract.Greeks.Delta, contract.Expiry - timedelta(self.liquidation_buffer), contract.Expiry, interest_rate, contract.LastPrice, contract.UnderlyingLastPrice, contract.Strike, current_datetime_naive)
            curr_predicted_profit = np.mean(curr_predicted_profits)
            curr_profits[contract] = curr_predicted_profit
        worst_current_contract = None
        if curr_profits:
            worst_current_contract = min(curr_profits, key=curr_profits.get)
        return worst_current_contract

    def rebalance_and_check_margin(self, data: Slice):
        interest_rate = self.risk_free_interest_rate_model.get_interest_rate(self.Time)
        current_datetime_naive = datetime(self.Time.year, self.Time.month, self.Time.day, self.Time.hour, self.Time.minute, self.Time.second)
        if (self.is_margin_low()):
            self.Debug("Margin too low, liquidating options")
            while(self.is_margin_low()):
                worst_option = self.get_worst_current_option(interest_rate, current_datetime_naive)
                if worst_option is None:
                    break
                self.MarketOrder(worst_option.Symbol, -1)
                del self.myOptions[worst_option.Symbol]
                del self.myOptionsDeltas[worst_option.Symbol]
                worst_option = self.get_worst_current_option(interest_rate, current_datetime_naive)
        self.delta_balance_portfolio(data)
    
    def is_margin_low(self):
        # potential_max_short_quantity  = sum([0.5 * 100 * self.Portfolio[option.Symbol].Quantity for option in self.myOptions.values()]) * self.margin_buffer
        # current_short_quantity = self.Portfolio["SPY"].Quantity * -1
        # if self.get_shortable_quantity() < potential_max_short_quantity - current_short_quantity:
        #     self.Debug(f"Margin too low! Shortable Quantity: {self.get_shortable_quantity()}, Potential Short Position: {potential_max_short_quantity - current_short_quantity}")
        #     return True
        # return False

        # if self.Portfolio.MarginRemaining < self.Portfolio.TotalPortfolioValue * self.margin_buffer:
        #     self.Debug(f"Margin too low! Margin Remaining: {self.Portfolio.MarginRemaining}, Total Portfolio Value: {self.Portfolio.TotalPortfolioValue}")
        #     return True
        # return False

        if self.Portfolio.TotalPortfolioValue < self.Portfolio[self.spy].Quantity * -1 * self.Portfolio[self.spy].Price * 1.2:
            self.Debug(f"Margin too low! Portfolio Value: {self.Portfolio.TotalPortfolioValue}, Spy Dollar Value: {self.Portfolio[self.spy].Quantity * self.Securities[self.spy].Price}")
            return True
        else: 
            self.Debug(f"Margin is good! Portfolio Value: {self.Portfolio.TotalPortfolioValue}, Spy Dollar Value: {self.Portfolio[self.spy].Quantity * self.Securities[self.spy].Price}")
            return False

        # free_margin = self.Securities["SPY"].BuyingPowerModel.GetMarginRemaining(self.Portfolio, self.Securities["SPY"], OrderDirection.Sell)
        # total_margin_used = self.Portfolio.TotalMarginUsed
        # if free_margin < total_margin_used * self.margin_buffer:
        #     self.Debug(f"Margin too low! Free Margin: {free_margin}, Total Margin Used: {total_margin_used}")
        #     return True
        # else:
        #     self.Debug(f"Margin is good! Free Margin: {free_margin}, Total Margin Used: {total_margin_used}")
        #     return False

    def get_shortable_quantity(self):
        #  return self.Portfolio.MarginRemaining // self.Securities["SPY"].Price
        # spy = self.Securities["SPY"]
        # free_margin = spy.BuyingPowerModel.GetBuyingPower(InsightDirection.Down)
        free_margin = self.Securities["SPY"].BuyingPowerModel.GetMarginRemaining(self.Portfolio, self.Securities["SPY"], OrderDirection.Sell)
        spy_price = self.Securities["SPY"].Price
        # margin_req = self.Securities["SPY"].MarginModel.GetInitialMarginRequirement(self.Securities["SPY"])
        margin_req = 0.5
        max_num_shorted_shares = free_margin / (margin_req * spy_price)
        return max_num_shorted_shares * (1 - self.margin_buffer)
    
    
    def on_end_of_algorithm(self):
        self.SetHoldings("SPY", 0)
        self.Liquidate()
        return super().on_end_of_algorithm()