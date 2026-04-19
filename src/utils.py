import numpy as np
from scipy.stats import norm

def validate_inputs(S, K, T, sigma):
    if S <= 0:
        raise ValueError("Spot price must be positive")
    if K <= 0:
        raise ValueError("Strike price must be positive")
    if T <= 0:
        raise ValueError("Maturity must be positive")
    if sigma <= 0:
        raise ValueError("Volatility must be positive")

def confidence_interval(samples, alpha=0.95):
    """
    Compute confidence interval for sample mean.

    Parameters
    ----------
    samples : array-like
        Monte Carlo sample values.
    alpha : float
        Confidence level (e.g. 0.95 for 95%).

    Returns
    -------
    mean : float
    lower : float
    upper : float
    """

    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")

    if len(samples) < 2:
        raise ValueError("need at least two samples")

    samples = np.asarray(samples)

    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    n = len(samples)

    z = norm.ppf(0.5 + alpha / 2)

    margin = z * std / np.sqrt(n)

    return mean, mean - margin, mean + margin