from scipy.optimize import newton
from black_scholes import call_price
from greeks import vega

def implied_vol(market_price, S, K, T, r, initial_guess=0.2, q=0.0):
    f = lambda sigma: call_price(S, K, T, r, sigma, q) - market_price
    f_prime = lambda sigma: vega(S, K, T, r, sigma, q)
    
    return newton(f, initial_guess, fprime=f_prime)
