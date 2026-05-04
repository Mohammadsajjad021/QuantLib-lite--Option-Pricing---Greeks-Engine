from scipy.stats import norm
import numpy as np
from black_scholes import d1, d2

def delta_call(S, K, T, r, sigma, q=0.0):
    return np.exp(-q * T) * norm.cdf(d1(S, K, T, r, sigma, q))

def delta_put(S, K, T, r, sigma, q=0.0):
    return np.exp(-q * T) * (norm.cdf(d1(S, K, T, r, sigma, q)) - 1)

def gamma(S, K, T, r, sigma, q=0.0):
    return np.exp(-q * T) * norm.pdf(d1(S, K, T, r, sigma, q))/(sigma*np.sqrt(T)*S)

def vega(S, K, T, r, sigma, q=0.0):
    return S * np.exp(-q * T) * norm.pdf(d1(S, K, T, r, sigma, q)) * np.sqrt(T)

def theta_call(S, K, T, r, sigma, q=0.0):
    D1 = d1(S, K, T, r, sigma, q)
    D2 = d2(S, K, T, r, sigma, q)
    return -S * np.exp(-q * T) * norm.pdf(D1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(D2) + q * S * np.exp(-q * T) * norm.cdf(D1)

def theta_put(S, K, T, r, sigma, q=0.0):
    D1 = d1(S, K, T, r, sigma, q)
    D2 = d2(S, K, T, r, sigma, q)
    return -S * np.exp(-q * T) * norm.pdf(D1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-D2) - q * S * np.exp(-q * T) * norm.cdf(-D1)

def rho_call(S, K, T, r, sigma, q=0.0):
    return K * T * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma, q))

def rho_put(S, K, T, r, sigma, q=0.0):
    return - K * T * np.exp(-r * T) * norm.cdf(-1 * d2(S, K, T, r, sigma, q))

def vanna(S,K,T,r,sigma,q=0.0):
    D1 = d1(S,K,T,r,sigma,q)
    D2 = d2(S,K,T,r,sigma,q)
    return -np.exp(-q * T) * norm.pdf(D1) * D2 / sigma

def vomma(S,K,T,r,sigma,q=0.0):
    D1 = d1(S,K,T,r,sigma,q)
    D2 = d2(S,K,T,r,sigma,q)
    vg = vega(S,K,T,r,sigma,q)
    return vg * D1 * D2 / sigma

def charm_call(S,K,T,r,sigma,q=0.0):
    D1 = d1(S,K,T,r,sigma,q)
    D2 = d2(S,K,T,r,sigma,q)
    num = np.exp(-q * T) * norm.pdf(D1) * (2*(r - q)*T - D2*sigma*np.sqrt(T))
    den = 2*T*sigma*np.sqrt(T)
    return q * np.exp(-q * T) * norm.cdf(D1) - num / den

def charm(S,K,T,r,sigma,q=0.0):
    D1 = d1(S,K,T,r,sigma,q)
    D2 = d2(S,K,T,r,sigma,q)
    return q * np.exp(-q * T) * norm.cdf(D1) - np.exp(-q * T) * norm.pdf(D1) * (2*(r - q)*T - D2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
