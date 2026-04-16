def test_call_bounds():
    from black_scholes import call_price
    import numpy as np

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    C = call_price(S, K, T, r, sigma)

    lower = max(0, S - K * np.exp(-r * T))
    upper = S

    assert lower <= C <= upper