def test_asian_call_positive():
    from monte_carlo import monte_carlo_asian_call

    price = monte_carlo_asian_call(100, 100, 1, 0.05, 0.2, n_sim=50000)

    assert price > 0


def test_asian_call_less_than_european_call():
    from black_scholes import call_price
    from monte_carlo import monte_carlo_asian_call

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    asian = monte_carlo_asian_call(S, K, T, r, sigma, n_sim=100000)
    european = call_price(S, K, T, r, sigma)

    assert asian < european
