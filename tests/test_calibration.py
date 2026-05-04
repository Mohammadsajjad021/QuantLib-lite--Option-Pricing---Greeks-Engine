import pytest


def test_implied_vol_from_market_call_price():
    from black_scholes import call_price
    from calibration import implied_vol_from_market_price

    S, K, T, r, sigma = 100, 105, 1.5, 0.03, 0.24
    market_price = call_price(S, K, T, r, sigma)

    calibrated = implied_vol_from_market_price(market_price, S, K, T, r)

    assert abs(calibrated - sigma) < 1e-6


def test_implied_vol_from_market_put_price():
    from black_scholes import put_price
    from calibration import implied_vol_from_market_price

    S, K, T, r, sigma = 100, 95, 0.75, 0.04, 0.31
    market_price = put_price(S, K, T, r, sigma)

    calibrated = implied_vol_from_market_price(
        market_price, S, K, T, r, option_type="put"
    )

    assert abs(calibrated - sigma) < 1e-6


def test_calibrate_implied_vol_surface_from_dict_quotes():
    from black_scholes import call_price, put_price
    from calibration import calibrate_implied_vols

    S, r, sigma = 100, 0.05, 0.22
    quotes = [
        {"K": 90, "T": 0.5, "market_price": call_price(S, 90, 0.5, r, sigma)},
        {"K": 100, "T": 1.0, "market_price": call_price(S, 100, 1.0, r, sigma)},
        {
            "K": 110,
            "T": 1.5,
            "market_price": put_price(S, 110, 1.5, r, sigma),
            "type": "put",
        },
    ]

    surface = calibrate_implied_vols(quotes, S, r)

    assert len(surface) == 3
    assert all(abs(row["implied_vol"] - sigma) < 1e-6 for row in surface)


def test_calibrate_flat_volatility_recovers_market_sigma():
    from black_scholes import call_price, put_price
    from calibration import MarketQuote, calibrate_flat_volatility

    S, r, sigma = 100, 0.03, 0.27
    quotes = [
        MarketQuote(90, 0.5, call_price(S, 90, 0.5, r, sigma)),
        MarketQuote(100, 1.0, call_price(S, 100, 1.0, r, sigma)),
        MarketQuote(110, 1.25, put_price(S, 110, 1.25, r, sigma), "put"),
    ]

    result = calibrate_flat_volatility(quotes, S, r)

    assert abs(result["sigma"] - sigma) < 1e-4
    assert result["rmse"] < 1e-4


def test_calibration_recovers_sigma_with_dividend_yield():
    from black_scholes import call_price
    from calibration import calibrate_flat_volatility, implied_vol_from_market_price

    S, K, T, r, sigma, q = 100, 100, 1, 0.05, 0.24, 0.03
    market_price = call_price(S, K, T, r, sigma, q)

    implied = implied_vol_from_market_price(market_price, S, K, T, r, q=q)
    result = calibrate_flat_volatility(
        [{"K": K, "T": T, "market_price": market_price}],
        S,
        r,
        q=q,
    )

    assert abs(implied - sigma) < 1e-6
    assert abs(result["sigma"] - sigma) < 1e-4


def test_market_price_outside_no_arbitrage_bounds_raises():
    from calibration import implied_vol_from_market_price

    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_vol_from_market_price(150, 100, 100, 1, 0.05)


def test_put_market_price_below_dividend_adjusted_bound_raises():
    from calibration import implied_vol_from_market_price
    import numpy as np

    S, K, T, r, q = 100, 120, 1, 0.01, 0.05
    lower_bound = K * np.exp(-r * T) - S * np.exp(-q * T)

    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_vol_from_market_price(
            lower_bound - 0.01,
            S,
            K,
            T,
            r,
            option_type="put",
            q=q,
        )
