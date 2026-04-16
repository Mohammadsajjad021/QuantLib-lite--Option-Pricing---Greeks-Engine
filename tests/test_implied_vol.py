def test_implied_vol_recovery():
    from black_scholes import call_price
    from implied_vol import implied_vol

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    market_price = call_price(S, K, T, r, sigma)

    iv = implied_vol(market_price, S, K, T, r)

    assert abs(iv - sigma) < 1e-3

def test_price_increases_with_vol():
    from black_scholes import call_price

    price_low = call_price(100, 100, 1, 0.05, 0.1)
    price_high = call_price(100, 100, 1, 0.05, 0.3)

    assert price_high > price_low