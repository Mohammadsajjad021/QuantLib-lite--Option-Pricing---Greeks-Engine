from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from black_scholes import call_price, put_price


@dataclass(frozen=True)
class MarketQuote:
    strike: float
    maturity: float
    price: float
    option_type: str = "call"


def _option_type(option_type):
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return option_type


def _quote_value(quote, *names):
    if isinstance(quote, dict):
        for name in names:
            if name in quote:
                return quote[name]
    else:
        for name in names:
            if hasattr(quote, name):
                return getattr(quote, name)
    raise ValueError(f"quote is missing required field: {names[0]}")


def _coerce_quote(quote):
    strike = float(_quote_value(quote, "strike", "K"))
    maturity = float(_quote_value(quote, "maturity", "T"))
    price = float(_quote_value(quote, "price", "market_price"))

    try:
        option_type = _quote_value(quote, "option_type", "type")
    except ValueError:
        option_type = "call"

    return MarketQuote(strike, maturity, price, _option_type(option_type))


def _validate_market_inputs(S, quote):
    if S <= 0:
        raise ValueError("Spot price must be positive")
    if quote.strike <= 0:
        raise ValueError("Strike price must be positive")
    if quote.maturity <= 0:
        raise ValueError("Maturity must be positive")
    if quote.price <= 0:
        raise ValueError("Market price must be positive")


def _price(S, K, T, r, sigma, option_type):
    if option_type == "call":
        return call_price(S, K, T, r, sigma)
    return put_price(S, K, T, r, sigma)


def _price_bounds(S, K, T, r, option_type):
    discounted_strike = K * np.exp(-r * T)
    if option_type == "call":
        return max(0.0, S - discounted_strike), S
    return max(0.0, discounted_strike - S), discounted_strike


def implied_vol_from_market_price(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    vol_lower=1e-6,
    vol_upper=5.0,
    tol=1e-8,
):
    """
    Calibrate Black-Scholes implied volatility from one market option price.
    """
    option_type = _option_type(option_type)
    quote = MarketQuote(K, T, market_price, option_type)
    _validate_market_inputs(S, quote)

    lower_price, upper_price = _price_bounds(S, K, T, r, option_type)
    if market_price < lower_price - tol or market_price > upper_price + tol:
        raise ValueError("Market price violates no-arbitrage bounds")

    def objective(sigma):
        return _price(S, K, T, r, sigma, option_type) - market_price

    low_error = objective(vol_lower)
    high_error = objective(vol_upper)

    if abs(low_error) < tol:
        return vol_lower
    if abs(high_error) < tol:
        return vol_upper
    if low_error * high_error > 0:
        raise ValueError("Could not bracket implied volatility")

    return brentq(objective, vol_lower, vol_upper, xtol=tol)


def calibrate_implied_vols(quotes, S, r, **kwargs):
    """
    Calibrate an implied volatility for each market quote.

    Quotes can be MarketQuote instances or dictionaries with strike/maturity/price
    fields. Dicts may also use K, T, and market_price aliases.
    """
    results = []
    for quote in quotes:
        quote = _coerce_quote(quote)
        _validate_market_inputs(S, quote)
        implied_vol = implied_vol_from_market_price(
            quote.price,
            S,
            quote.strike,
            quote.maturity,
            r,
            quote.option_type,
            **kwargs,
        )
        results.append(
            {
                "strike": quote.strike,
                "maturity": quote.maturity,
                "option_type": quote.option_type,
                "market_price": quote.price,
                "implied_vol": implied_vol,
            }
        )
    return results


def calibrate_flat_volatility(quotes, S, r, vol_lower=1e-6, vol_upper=5.0):
    """
    Calibrate one Black-Scholes volatility to a set of market option prices.
    """
    quotes = [_coerce_quote(quote) for quote in quotes]
    if not quotes:
        raise ValueError("At least one market quote is required")

    for quote in quotes:
        _validate_market_inputs(S, quote)

    def squared_error(sigma):
        errors = [
            _price(S, q.strike, q.maturity, r, sigma, q.option_type) - q.price
            for q in quotes
        ]
        return float(np.sum(np.square(errors)))

    calibration = minimize_scalar(
        squared_error,
        bounds=(vol_lower, vol_upper),
        method="bounded",
    )
    if not calibration.success:
        raise RuntimeError("Flat volatility calibration failed")

    sigma = float(calibration.x)
    errors = [
        _price(S, q.strike, q.maturity, r, sigma, q.option_type) - q.price
        for q in quotes
    ]
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    return {
        "sigma": sigma,
        "rmse": rmse,
        "pricing_errors": errors,
    }
