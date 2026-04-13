from scipy.optimize import newton
from black_scholes import call_price
from greeks import vega

def implied_vol(market_price, S, K, T, r, initial_guess=0.2):
    f = lambda sigma: call_price(S, K, T, r, sigma) - market_price

    return newton(f, initial_guess)