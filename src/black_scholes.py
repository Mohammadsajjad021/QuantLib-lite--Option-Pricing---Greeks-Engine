import numpy as np
from scipy.stats import norm

def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def call_price(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    call_price = S * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)
    return call_price

def put_price(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    put_price = K * np.exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)
    return put_price