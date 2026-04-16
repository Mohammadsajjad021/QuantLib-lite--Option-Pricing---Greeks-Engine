def test_call_price_known_value():
    from black_scholes import call_price

    price = call_price(100, 100, 1, 0.05, 0.2)

    assert abs(price - 10.45) < 0.1


def test_put_call_parity():
    from black_scholes import call_price, put_price
    import numpy as np

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    C = call_price(S, K, T, r, sigma)
    P = put_price(S, K, T, r, sigma)

    lhs = C - P
    rhs = S - K * np.exp(-r * T)

    assert abs(lhs - rhs) < 1e-6