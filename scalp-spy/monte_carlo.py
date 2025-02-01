import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime, timedelta
from scipy.stats import norm
import numpy as np


def gamma_scalping_projection(iv, initial_delta, liquidation_date, expiry_date, risk_free_rate, initial_option_price, initial_underlying_price, strike_price, todays_date):
    """
    Run a Monte Carlo simulation for gamma scalping to predict the profit of entering a position.
    
    Parameters:
        iv (float): Implied volatility of the option
        initial_delta (float): Initial delta of the option
        liquidation_date (datetime): Date of liquidation of the position
        expiry_date (datetime): Expiry date of the option
        risk_free_rate (float): Risk-free interest rate
        initial_option_price (float): Price of the option
        underlying_price (float): Current price of the underlying asset
        strike_price (float): Strike price of the option

    Returns:
        float: Net profit from gamma scalping
    """
    # Initialize variables
    total_portfolio_value = -initial_option_price  # Initial portfolio value = Option price * number of contracts * 100 (for 100 shares per contract)
    current_underlying_price = initial_underlying_price
    current_option_price = initial_option_price
    days_to_expiry = (expiry_date - todays_date).days  # Remaining time to expiry in days
    trading_days_until_liquidation = get_market_open_days(todays_date, liquidation_date)
    trading_hours_until_liquidation = trading_days_until_liquidation * 6 
    initial_position = calculate_underlying_position_adjustment_for_delta_neutrality(initial_delta, 0)
    current_position = initial_position
    total_portfolio_value += initial_position * current_underlying_price  # Adjust portfolio value based on initial position

    hourly_volatility = iv / np.sqrt(6.5 * 252)  # Convert daily volatility to hourly volatility

    for _ in range(trading_hours_until_liquidation):
        # Simulate price movement
        dS = current_underlying_price * hourly_volatility * np.sqrt(1 / 252) * np.random.normal()  # dS = S * volatility * sqrt(dt) * Z
        current_underlying_price += dS

        # Check if ITM
        if current_underlying_price > strike_price:
            break

        # Update time to expiry
        days_to_expiry = (expiry_date - todays_date).days

        # Update delta 
        current_delta = calculate_delta(current_underlying_price, strike_price, days_to_expiry / 252, iv, risk_free_rate)

        # Update theta, current option price
        current_theta = calculate_theta(current_underlying_price, strike_price, days_to_expiry / 252, iv, risk_free_rate)
        current_option_price -= current_theta / 6  # Update option price every hour

        # Adjust the position to remain delta neutral
        position_adjustment = calculate_underlying_position_adjustment_for_delta_neutrality(current_delta, current_position)
        current_position += position_adjustment
        total_portfolio_value += position_adjustment * current_underlying_price  # Adjust portfolio value based on position change
    
    # Liquidate the position
    total_portfolio_value += current_option_price
    total_portfolio_value -= current_position * current_underlying_price

    return total_portfolio_value

# def gamma_scalping_projection(underlying_price, iv, gamma, theta, num_contracts=1, days=21):
    """
    Simulate the profit from gamma scalping over a period of days.
    underlying_price: Current price of the underlying asset
    iv: Implied volatility of the option
    gamma: Option's gamma
    theta: Option's theta
    num_contracts: Number of option contracts
    days: Number of days to simulate
    """

    trading_days_until_expiry = get_market_open_days(days)
    trading_hours_until_expiry = trading_days_until_expiry * 6.5  # 6.5 hours per trading day

    daily_volatility = iv / np.sqrt(252)     # Convert annualized IV to daily volatility
    hourly_volatility = daily_volatility / np.sqrt(6.5)  # Convert daily volatility to hourly volatility
    
    # Initialize variables to accumulate gains and losses
    total_gamma_gain = 0
    total_theta_decay = 0
    total_delta_hedging_cost = 0

    last_price = underlying_price
    delta = 0  # Initial delta

    
    for _ in range(int(trading_hours_until_expiry)):
        # Simulate daily price movement (delta S) as a random draw from normal distribution
        hourly_price_change = np.random.normal(0, hourly_volatility)
        new_price = last_price + hourly_price_change
        last_price += hourly_price_change

        # Calculate daily gamma gain: 0.5 * gamma * (delta S)^2
        hourly_gamma_gain = 0.5 * gamma * (hourly_price_change ** 2) * num_contracts  # Multiply by num_contracts
        total_gamma_gain += hourly_gamma_gain
        
        # Calculate daily theta decay
        daily_theta = theta / 252 * num_contracts  # Multiply by num_contracts
        total_theta_decay += daily_theta

    # Net profit from gamma scalping over the period
    net_profit = total_gamma_gain - total_theta_decay
    return net_profit

def monte_carlo_gamma_scalping(iv, initial_delta, liquidation_date, expiry_date, risk_free_rate, initial_option_price, initial_underlying_price, strike_price, todays_date = datetime.now(), simulations=1000):
    """
    Run a Monte Carlo simulation to project the profit from gamma scalping over a period of days.
    
    Parameters:
        iv (float): Implied volatility of the option
        initial_delta (float): Initial delta of the option
        liquidation_date (datetime): Date of liquidation of the position
        expiry_date (datetime): Expiry date of the option
        risk_free_rate (float): Risk-free interest rate
        initial_option_price (float): Price of the option
        underlying_price (float): Current price of the underlying asset
        strike_price (float): Strike price of the option
        simulations (int): Number of Monte Carlo simulations to run

    Returns:
        list: List of projected profits from gamma scalping
    """

    # Run the projection multiple times and store the results
    results = []
    
    for _ in range(simulations):
        profit = gamma_scalping_projection(iv, initial_delta, liquidation_date, expiry_date, risk_free_rate, initial_option_price, initial_underlying_price, strike_price, todays_date)
        results.append(profit)
    
    return results

def get_market_open_days(start_date, end_date):
    # Create a date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' frequency excludes weekends

    # Get US Federal Holidays
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)

    # Exclude holidays from the date range
    market_open_days = date_range.difference(holidays)

    return len(market_open_days)

def calculate_delta(S, K, T, sigma, r):
    """
    Calculate the delta of a European option using the Black-Scholes model.
    
    Parameters:
        S (float): Current price of the underlying asset
        K (float): Strike price of the option
        T (float): Time to expiration (in years)
        sigma (float): Implied volatility of the option
        r (float): Risk-free interest rate (annualized)
    
    Returns:
        float: Delta of the option
    """
    # Calculate d1 from the Black-Scholes formula
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    delta = norm.cdf(d1)  # CDF for call options
    
    return delta

def calculate_gamma(S, K, T, sigma, r):
    """
    Calculate the gamma of a European option using the Black-Scholes model.
    
    Parameters:
        S (float): Current price of the underlying asset
        K (float): Strike price of the option
        T (float): Time to expiration (in years)
        sigma (float): Implied volatility of the option
        r (float): Risk-free interest rate (annualized)
    
    Returns:
        float: Gamma of the option
    """
    # Calculate d1 from the Black-Scholes formula
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Calculate Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    return gamma

def calculate_theta(S, K, T, sigma, r):
    """
    Calculate the theta of a European option using the Black-Scholes model.
    
    Parameters:
        S (float): Current price of the underlying asset
        K (float): Strike price of the option
        T (float): Time to expiration (in years)
        sigma (float): Implied volatility of the option
        r (float): Risk-free interest rate (annualized)
    
    Returns:
        float: Theta of the option
    """
    # Calculate d1 and d2 from the Black-Scholes formula
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate the standard normal PDF and CDF values
    pdf_d1 = norm.pdf(d1)  # Normal PDF evaluated at d1
    cdf_d2 = norm.cdf(d2)  # Normal CDF evaluated at d2
    cdf_d1 = norm.cdf(d1)  # Normal CDF evaluated at d1

    theta = - (S * sigma * pdf_d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2
    
    return theta

def calculate_underlying_position_adjustment_for_delta_neutrality(option_delta, current_num_underlying_shares):
    """
    Calculate the number of shares to buy/sell to maintain delta neutrality.
    
    Parameters:
        option_delta (float): Delta of the option
        current_num_underlying_shares (int): Current number of shares of the underlying asset
    
    Returns:
        int: Number of shares to buy/sell to maintain delta neutrality
    """
    # Calculate the number of shares to buy/sell to maintain delta neutrality
    option_total_delta = option_delta * 100
    portfolio_total_delta = option_total_delta + current_num_underlying_shares
    
    return round(-portfolio_total_delta)

# Example inputs
iv = 0.148401292935819  # Implied volatility (25%)
initial_delta = 0.433563103239395  # Initial delta 
liquidation_date = datetime.today() + timedelta(days=28)  # Liquidate in 7 days
expiry_date = datetime.today() + timedelta(days=30)  # Option expires in 30 days
risk_free_rate = 0.01  # Risk-free interest rate (3%)
initial_option_price = 5.0  # Initial option price ($5 per share)
initial_underlying_price = 100.0  # Current price of the underlying asset ($100)
strike_price = 110.0  # Strike price of the option ($100)

# Call the function
profit = monte_carlo_gamma_scalping(
    iv=iv,
    initial_delta=initial_delta,
    liquidation_date=liquidation_date,
    expiry_date=expiry_date,
    risk_free_rate=risk_free_rate,
    initial_option_price=initial_option_price,
    initial_underlying_price=initial_underlying_price,
    strike_price=strike_price
)

print(f"Projected profit: {np.mean(profit)}")