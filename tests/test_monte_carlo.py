def test_mc_matches_black_scholes():
    from black_scholes import call_price
    from monte_carlo import monte_carlo_call

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    bs = call_price(S, K, T, r, sigma)
    mc = monte_carlo_call(S, K, T, r, sigma, n_sim=200000)

    assert abs(bs - mc) < 0.5

def test_mc_put_positive():
    from monte_carlo import monte_carlo_put

    price = monte_carlo_put(100, 100, 1, 0.05, 0.2)

    assert price > 0