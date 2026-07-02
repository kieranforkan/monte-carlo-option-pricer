import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def estimate_gbm(prices, dt):
    """
    Estimate the drift and volatility of a Geometric Brownian Motion (GBM)
    from a series of stock prices.

    Parameters:
    - prices: array-like, stock prices over time
    - dt: float, time increment between prices

    Returns:
    - mu: estimated drift
    - sigma: estimated volatility
    """
    log_returns = np.diff(np.log(prices))
    sigma = np.std(log_returns) / np.sqrt(dt)
    mu = np.mean(log_returns) / dt + 0.5 * sigma**2
    return mu, sigma

def estimate_ou(X, dt):
    X = np.asarray(X)
    X_curr = X[:-1]   # X[t]     (all but last)
    X_next = X[1:]    # X[t+1]   (all but first)

    def neg_log_likelihood(params):
        theta, mu, sigma = params
        if theta <= 0 or sigma <= 0:
            return 1e10  # reject invalid parameters

        mean = mu + (X_curr - mu) * np.exp(-theta * dt)
        var = sigma**2 / (2 * theta) * (1 - np.exp(-2 * theta * dt))
        sd = np.sqrt(var)

        # log-likelihood of each step under its Normal, summed
        ll = np.sum(norm.logpdf(X_next, loc=mean, scale=sd))
        return -ll   # negative because we minimise

    # initial guess, then optimise
    guess = [1.0, np.mean(X), np.std(np,diff(X)) / np.sqrt(dt)]
    bounds = [(1e-5, None), (None, None), (1e-6, None)] # theta > 0, sigma > 0, mu free
    result = minimize(neg_log_likelihood, guess, method="L-BFGS-B")
    theta_hat, mu_hat, sigma_hat = result.x
    return theta_hat, mu_hat, sigma_hat