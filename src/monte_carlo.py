import numpy as np


def simulate_terminal_price(S0, T, r, sigma, n_sim):
    Z = np.random.standard_normal(n_sim)

    drift = r - 0.5 * sigma**2 * T
    vol = sigma * np.sqrt(T)
    
    return S0 * np.exp(drift + vol * Z)

def simulate_terminal_price_anithetic(S0, T, r, sigma, n_sim):
    n_half = n_sim // 2
    Z = np.random.standard_normal(n_half)

    drift = (r - 0.5*sigma**2)*T
    vol = sigma*np.sqrt(T)

    ST1 = S0*np.exp(drift + vol*Z)
    ST2 = S0*np.exp(drift - vol*Z)

    return ST1, ST2

def monte_carlo_call(S0, K, T, r, sigma, n_sim=100000):
    ST = simulate_terminal_price(S0, T, r, sigma, n_sim)

    payoff = np.maximum(ST - K, 0)

    return np.exp(-r * T) * np.mean(payoff)


def monte_carlo_put(S0, K, T, r, sigma, n_sim=100000):
    ST = simulate_terminal_price(S0, T, r, sigma, n_sim)

    payoff = np.maximum(K - ST, 0)

    return np.exp(-r * T) * np.mean(payoff)


def monte_carlo_call_antithetic(S0,K,T,r,sigma,n_sim=100000):
    ST1 , ST2 = simulate_terminal_price_anithetic(S0, T, r, sigma, n_sim)
    
    payoff1 = np.maximum(ST1-K,0)
    payoff2 = np.maximum(ST2-K,0)

    payoff = 0.5*(payoff1 + payoff2)

    return np.exp(-r*T)*np.mean(payoff)

def monte_carlo_put_antithetic(S0,K,T,r,sigma,n_sim=100000):
    ST1 , ST2 = simulate_terminal_price_anithetic(S0, T, r, sigma, n_sim)
    
    payoff1 = np.maximum(K-ST1,0)
    payoff2 = np.maximum(K-ST2,0)

    payoff = 0.5*(payoff1 + payoff2)

    return np.exp(-r*T)*np.mean(payoff)

def monte_carlo_call_control(S0,K,T,r,sigma,n_sim=100000):
    ST = simulate_terminal_price(S0, T, r, sigma, n_sim)

    X = np.maximum(ST-K,0)      
    Y = ST                     

    EY = S0*np.exp(r*T)

    cov_xy = np.cov(X,Y, ddof=1)[0,1]
    var_y = np.var(Y, ddof=1)

    b = cov_xy / var_y

    X_cv = X - b*(Y - EY)

    return np.exp(-r*T)*np.mean(X_cv)

def monte_carlo_put_control(S0,K,T,r,sigma,n_sim=100000):
    ST = simulate_terminal_price(S0, T, r, sigma, n_sim)

    X = np.maximum(K-ST,0)      
    Y = ST                     

    EY = S0*np.exp(r*T)

    cov_xy = np.cov(X,Y, ddof=1)[0,1]
    var_y = np.var(Y, ddof=1)

    b = cov_xy / var_y

    X_cv = X - b*(Y - EY)

    return np.exp(-r*T)*np.mean(X_cv)