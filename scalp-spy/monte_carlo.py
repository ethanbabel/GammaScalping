import numpy as np
import matplotlib.pyplot as plt


def gamma_scalping_projection(underlying_price, iv, gamma, theta, num_contracts=1, days=21):
    """
    Simulate the profit from gamma scalping over a period of days.
    underlying_price: Current price of the underlying asset
    iv: Implied volatility of the option
    gamma: Option's gamma
    theta: Option's theta
    num_contracts: Number of option contracts
    days: Number of days to simulate
    """

    daily_volatility = iv / np.sqrt(252)     # Convert annualized IV to daily volatility
    
    # Initialize variables to accumulate gains and losses
    total_gamma_gain = 0
    total_theta_decay = 0
    
    for _ in range(days):
        # Simulate daily price movement (delta S) as a random draw from normal distribution
        daily_price_change = np.random.normal(0, daily_volatility * underlying_price)
        
        # Calculate daily gamma gain: 0.5 * gamma * (delta S)^2
        daily_gamma_gain = 0.5 * gamma * (daily_price_change ** 2) * num_contracts * 100  # 100 shares per contract
        total_gamma_gain += daily_gamma_gain
        
        # Calculate daily theta decay
        daily_theta = theta / 252 * num_contracts  # Multiply by num_contracts
        total_theta_decay += daily_theta

    # Net profit from gamma scalping over the period
    net_profit = total_gamma_gain - total_theta_decay
    return net_profit

def monte_carlo_gamma_scalping(underlying_price, iv, gamma, theta, num_contracts=1, days=21, simulations=10000):
    """
    Run a Monte Carlo simulation to project the profit from gamma scalping over a period of days.
    underlying_price: Current price of the underlying asset
    iv: Implied volatility of the option
    gamma: Option's gamma
    theta: Option's theta
    num_contracts: Number of option contracts
    days: Number of days to simulate
    simulations: Number of simulations to run
    """

    # Run the projection multiple times and store the results
    results = []
    
    for _ in range(simulations):
        profit = gamma_scalping_projection(underlying_price, iv, gamma, theta, num_contracts, days)
        results.append(profit)
    
    return results

# Example parameters
underlying_price = 100    # Underlying price
strike_price = 105        # Strike price
T = 30 / 365              # Time to expiration in years (e.g., 30 days)
r = 0.01                  # Risk-free interest rate
iv = 0.25                 # Implied volatility (e.g., 25%)
gamma = 0.02              # Option's gamma
theta = -0.05             # Option's theta (e.g., -5 dollars per year)
num_contracts = 10        # Number of option contracts
simulations = 10000       # Number of Monte Carlo simulations

# Run Monte Carlo simulation
results = monte_carlo_gamma_scalping(
    underlying_price, iv, gamma, theta, num_contracts=num_contracts, days=21, simulations=simulations
)

# Analyze results
mean_profit = np.mean(results)
std_profit = np.std(results)

print(f"Mean Projected Profit from Gamma Scalping: ${mean_profit:.2f}")
print(f"Standard Deviation of Projected Profit: ${std_profit:.2f}")

# Plot histogram of results
plt.hist(results, bins=50, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel("Projected Profit")
plt.ylabel("Frequency")
plt.title("Distribution of Projected Profits from Gamma Scalping (Monte Carlo)")
plt.savefig("gamma_scalping_histogram.png")


