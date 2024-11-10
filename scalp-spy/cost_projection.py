
def gamma_scalping_cost(garch_model, current_datetime, time_horizon, underlying_price, option_price, gamma, delta):
    """
    Calculates the initial cost of the gamma scalping position and the cost of future adjustments.
    
    garch_model: GARCH model object
    current_datetime: Current date and time
    time_horizon: Number of days to project
    underlying_price: Current price of the underlying asset
    option_price: Price of the option
    gamma: Option's gamma
    delta: Option's delta
    initial_short_position: Whether you're opening a new short position or adjusting
    """
    
    # Initial cost of the short position (this is the margin required)
    cost_of_short_position = underlying_price * 100 * delta  # 100 shares per contract

    # Project future price movements using GARCH model
    price_move = garch_model.predict(time_horizon=time_horizon, startdate=current_datetime)
    price_move_as_percent = price_move[time_horizon-1] / underlying_price * 100
    delta_change = gamma * price_move_as_percent
    cost_of_future_adjustments = delta_change * 100

    return option_price + cost_of_short_position + cost_of_future_adjustments

