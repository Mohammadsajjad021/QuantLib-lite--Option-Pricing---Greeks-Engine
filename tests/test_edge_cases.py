def test_zero_maturity():
    from black_scholes import call_price

    price = call_price(100, 90, 1e-8, 0.05, 0.2)

    assert abs(price - 10) < 1

def test_deep_in_the_money():
    from black_scholes import call_price

    price = call_price(200, 50, 1, 0.05, 0.2)

    assert price > 140