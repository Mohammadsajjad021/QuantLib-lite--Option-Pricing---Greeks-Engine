import numpy as np


def simulate_terminal_price(S0, T, r, sigma, n_sim):
    Z = np.random.standard_normal(n_sim)

    return S0 * np.exp(
        (r - 0.5 * sigma**2) * T
        + sigma * np.sqrt(T) * Z
    )


def monte_carlo_call(S0, K, T, r, sigma, n_sim=100000):
    ST = simulate_terminal_price(S0, T, r, sigma, n_sim)

    payoff = np.maximum(ST - K, 0)

    return np.exp(-r * T) * np.mean(payoff)


def monte_carlo_put(S0, K, T, r, sigma, n_sim=100000):
    ST = simulate_terminal_price(S0, T, r, sigma, n_sim)

    payoff = np.maximum(K - ST, 0)

    return np.exp(-r * T) * np.mean(payoff)