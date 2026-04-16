def test_delta_bounds():
    from greeks import delta_call

    delta = delta_call(100, 100, 1, 0.05, 0.2)

    assert 0 <= delta <= 1

def test_gamma_positive():
    from greeks import gamma

    g = gamma(100, 100, 1, 0.05, 0.2)

    assert g > 0

def test_vega_positive():
    from greeks import vega

    v = vega(100, 100, 1, 0.05, 0.2)

    assert v > 0

def test_delta_finite_difference():
    from black_scholes import call_price
    from greeks import delta_call

    S = 100
    eps = 1e-4

    price_up = call_price(S + eps, 100, 1, 0.05, 0.2)
    price_down = call_price(S - eps, 100, 1, 0.05, 0.2)

    numerical_delta = (price_up - price_down) / (2 * eps)
    analytical_delta = delta_call(S, 100, 1, 0.05, 0.2)

    assert abs(numerical_delta - analytical_delta) < 1e-3