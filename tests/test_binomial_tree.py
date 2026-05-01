def test_binomial_call_matches_black_scholes():
    from binomial_tree import binomial_call
    from black_scholes import call_price

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    tree_price = binomial_call(S, K, T, r, sigma, n_steps=1000)
    bs_price = call_price(S, K, T, r, sigma)

    assert abs(tree_price - bs_price) < 0.05


def test_binomial_put_matches_black_scholes():
    from binomial_tree import binomial_put
    from black_scholes import put_price

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    tree_price = binomial_put(S, K, T, r, sigma, n_steps=1000)
    bs_price = put_price(S, K, T, r, sigma)

    assert abs(tree_price - bs_price) < 0.05


def test_american_put_at_least_european_put():
    from binomial_tree import binomial_put

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    european = binomial_put(S, K, T, r, sigma, n_steps=500)
    american = binomial_put(S, K, T, r, sigma, n_steps=500, exercise="american")

    assert american >= european
