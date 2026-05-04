import numpy as np
from scipy.stats import norm

def simulate_terminal_price(S0, T, r, sigma, n_sim, q=0.0):
    Z = np.random.standard_normal(n_sim)

    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    
    return S0 * np.exp(drift + vol * Z)

def simulate_terminal_price_anithetic(S0, T, r, sigma, n_sim, q=0.0):
    n_half = n_sim // 2
    Z = np.random.standard_normal(n_half)

    drift = (r - q - 0.5*sigma**2)*T
    vol = sigma*np.sqrt(T)

    ST1 = S0*np.exp(drift + vol*Z)
    ST2 = S0*np.exp(drift - vol*Z)

    return ST1, ST2

def simulate_terminal_price_stratified(S0, T, r, sigma, n_sim, q=0.0):
    U = (np.arange(n_sim) + np.random.rand(n_sim)) / n_sim
    Z = norm.ppf(U)

    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    
    return S0 * np.exp(drift + vol * Z)

def simulate_gbm_paths(S0, T, r, sigma, n_sim, n_steps, q=0.0):
    dt = T / n_steps
    Z = np.random.standard_normal((n_sim, n_steps))
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)

    return S0 * np.exp(log_paths)

def monte_carlo_call_naive(S0, K, T, r, sigma, n_sim=100000, q=0.0):
    ST = simulate_terminal_price(S0, T, r, sigma, n_sim, q)

    payoff = np.maximum(ST - K, 0)

    return np.exp(-r * T) * np.mean(payoff)


def monte_carlo_put_naive(S0, K, T, r, sigma, n_sim=100000, q=0.0):
    ST = simulate_terminal_price(S0, T, r, sigma, n_sim, q)

    payoff = np.maximum(K - ST, 0)

    return np.exp(-r * T) * np.mean(payoff)


def monte_carlo_call_antithetic(S0,K,T,r,sigma,n_sim=100000,q=0.0):
    ST1 , ST2 = simulate_terminal_price_anithetic(S0, T, r, sigma, n_sim, q)
    
    payoff1 = np.maximum(ST1-K,0)
    payoff2 = np.maximum(ST2-K,0)

    payoff = 0.5*(payoff1 + payoff2)

    return np.exp(-r*T)*np.mean(payoff)

def monte_carlo_put_antithetic(S0,K,T,r,sigma,n_sim=100000,q=0.0):
    ST1 , ST2 = simulate_terminal_price_anithetic(S0, T, r, sigma, n_sim, q)
    
    payoff1 = np.maximum(K-ST1,0)
    payoff2 = np.maximum(K-ST2,0)

    payoff = 0.5*(payoff1 + payoff2)

    return np.exp(-r*T)*np.mean(payoff)

def monte_carlo_call_control(S0,K,T,r,sigma,n_sim=100000,q=0.0):
    ST = simulate_terminal_price(S0, T, r, sigma, n_sim, q)

    X = np.maximum(ST-K,0)      
    Y = ST                     

    EY = S0*np.exp((r - q)*T)

    cov_xy = np.cov(X,Y, ddof=1)[0,1]
    var_y = np.var(Y, ddof=1)

    b = cov_xy / var_y

    X_cv = X - b*(Y - EY)

    return np.exp(-r*T)*np.mean(X_cv)

def monte_carlo_put_control(S0,K,T,r,sigma,n_sim=100000,q=0.0):
    ST = simulate_terminal_price(S0, T, r, sigma, n_sim, q)

    X = np.maximum(K-ST,0)      
    Y = ST                     

    EY = S0*np.exp((r - q)*T)

    cov_xy = np.cov(X,Y, ddof=1)[0,1]
    var_y = np.var(Y, ddof=1)

    b = cov_xy / var_y

    X_cv = X - b*(Y - EY)

    return np.exp(-r*T)*np.mean(X_cv)

def monte_carlo_call_stratified(S0, K, T, r, sigma, n_sim=100000, q=0.0):
    ST = simulate_terminal_price_stratified(S0, T, r, sigma, n_sim, q)

    payoff = np.maximum(ST - K, 0)

    return np.exp(-r * T) * np.mean(payoff)


def monte_carlo_put_stratified(S0, K, T, r, sigma, n_sim=100000, q=0.0):
    ST = simulate_terminal_price_stratified(S0, T, r, sigma, n_sim, q)

    payoff = np.maximum(K - ST, 0)

    return np.exp(-r * T) * np.mean(payoff)

def monte_carlo_asian_call(S0, K, T, r, sigma, n_sim=100000, n_steps=252, q=0.0):
    paths = simulate_gbm_paths(S0, T, r, sigma, n_sim, n_steps, q)
    avg_price = np.mean(paths, axis=1)

    payoff = np.maximum(avg_price - K, 0)

    return np.exp(-r * T) * np.mean(payoff)


def monte_carlo_asian_put(S0, K, T, r, sigma, n_sim=100000, n_steps=252, q=0.0):
    paths = simulate_gbm_paths(S0, T, r, sigma, n_sim, n_steps, q)
    avg_price = np.mean(paths, axis=1)

    payoff = np.maximum(K - avg_price, 0)

    return np.exp(-r * T) * np.mean(payoff)

def monte_carlo_call(S0, K, T, r, sigma, n_sim=100000, mode='naive', q=0.0):
    match mode:
        case 'naive':
            return monte_carlo_call_naive(S0, K, T, r, sigma, n_sim=n_sim, q=q)
        case 'antithetic':
            return monte_carlo_call_antithetic(S0, K, T, r, sigma, n_sim=n_sim, q=q)
        case 'control':
            return monte_carlo_call_control(S0, K, T, r, sigma, n_sim=n_sim, q=q)
        case 'stratified':
            return monte_carlo_call_stratified(S0, K, T, r, sigma, n_sim=n_sim, q=q)

def monte_carlo_put(S0, K, T, r, sigma, n_sim=100000, mode='naive', q=0.0):
    match mode:
        case 'naive':
            return monte_carlo_put_naive(S0, K, T, r, sigma, n_sim=n_sim, q=q)
        case 'antithetic':
            return monte_carlo_put_antithetic(S0, K, T, r, sigma, n_sim=n_sim, q=q)
        case 'control':
            return monte_carlo_put_control(S0, K, T, r, sigma, n_sim=n_sim, q=q)
        case 'stratified':
            return monte_carlo_put_stratified(S0, K, T, r, sigma, n_sim=n_sim, q=q)
