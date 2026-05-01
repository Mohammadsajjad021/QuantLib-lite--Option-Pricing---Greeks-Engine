import numpy as np


def binomial_price(S, K, T, r, sigma, n_steps=100, option_type="call", exercise="european"):
    if S <= 0:
        raise ValueError("Spot price must be positive")
    if K <= 0:
        raise ValueError("Strike price must be positive")
    if T <= 0:
        raise ValueError("Maturity must be positive")
    if sigma <= 0:
        raise ValueError("Volatility must be positive")
    if n_steps <= 0:
        raise ValueError("Number of steps must be positive")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if exercise not in {"european", "american"}:
        raise ValueError("exercise must be 'european' or 'american'")

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    discount = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)

    if not (0 <= p <= 1):
        raise ValueError("Invalid risk-neutral probability; increase n_steps or check inputs")

    nodes = np.arange(n_steps + 1)
    stock_prices = S * (u ** nodes) * (d ** (n_steps - nodes))

    if option_type == "call":
        option_values = np.maximum(stock_prices - K, 0)
    else:
        option_values = np.maximum(K - stock_prices, 0)

    for step in range(n_steps - 1, -1, -1):
        option_values = discount * (p * option_values[1:] + (1 - p) * option_values[:-1])

        if exercise == "american":
            nodes = np.arange(step + 1)
            stock_prices = S * (u ** nodes) * (d ** (step - nodes))

            if option_type == "call":
                exercise_values = np.maximum(stock_prices - K, 0)
            else:
                exercise_values = np.maximum(K - stock_prices, 0)

            option_values = np.maximum(option_values, exercise_values)

    return option_values[0]


def binomial_call(S, K, T, r, sigma, n_steps=100, exercise="european"):
    return binomial_price(S, K, T, r, sigma, n_steps, option_type="call", exercise=exercise)


def binomial_put(S, K, T, r, sigma, n_steps=100, exercise="european"):
    return binomial_price(S, K, T, r, sigma, n_steps, option_type="put", exercise=exercise)
