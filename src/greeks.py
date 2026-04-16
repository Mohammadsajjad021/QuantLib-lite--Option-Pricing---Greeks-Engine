from scipy.stats import norm
import numpy as np
from black_scholes import d1, d2

def delta_call(S, K, T, r, sigma):
    return norm.cdf(d1(S, K, T, r, sigma))

def delta_put(S, K, T, r, sigma):
    return norm.cdf(d1(S, K, T, r, sigma)) - 1

def gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S, K, T, r, sigma))/(sigma*np.sqrt(T)*S)

def vega(S, K, T, r, sigma):
    return S * norm.pdf(d1(S, K, T, r, sigma)) * np.sqrt(T)

def theta_call(S, K, T, r, sigma):
    return -S * norm.pdf(d1(S, K, T, r, sigma)) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))

def theta_put(S, K, T, r, sigma):
    return -S * norm.pdf(d1(S, K, T, r, sigma)) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-1 * d2(S, K, T, r, sigma))

def rho_call(S, K, T, r, sigma):
    return K * T * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))

def rho_put(S, K, T, r, sigma):
    return - K * T * np.exp(-r * T) * norm.cdf(-1 * d2(S, K, T, r, sigma))