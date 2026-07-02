import numpy as np
from src.simulation import simulate_gbm
from scipy.stats import norm
from scipy.optimize import brentq

def price_european_call(S0, K, r, sigma, T, N, M):
    payoffs = np.zeros(M)          # One payoff per simulated path
    for i in range (M):
        path = simulate_gbm(S0, r, sigma, T, N)        # drift = r
        S_T = path[-1]
        payoffs[i] = np.maximum(S_T - K, 0)      # Call payoff
    discounted = np.exp(-r * T) * np.mean(payoffs)    # Average payoff
    return discounted

def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def price_asian_call(S0, K, r, sigma, T, N, M):
    payoffs = np.zeros(M)          # One payoff per simulated path
    for i in range (M):
        path = simulate_gbm(S0, r, sigma, T, N)        # drift = r
        avg_price = np.mean(path)      # Average price over the path
        payoffs[i] = np.maximum(avg_price - K, 0)      # Asian call payoff
    discounted = np.exp(-r * T) * np.mean(payoffs)    # Average payoff
    return discounted

def price_european_call_antithetic(S0, K, r, sigma, T, M):
    payoffs = np.zeros(M)
    for i in range(M):
        Z = np.random.normal()
        ST_plus = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        ST_minus = S0 * np.exp((r - 0.5 * sigma**2) * T - sigma * np.sqrt(T) * Z)
        payoff_plus = np.maximum(ST_plus - K, 0)
        payoff_minus = np.maximum(ST_minus - K, 0)
        payoffs[i] = 0.5 * (payoff_plus + payoff_minus)
    discounted = np.exp(-r * T) * np.mean(payoffs)
    return discounted

def bs_delta_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    return delta

def fd_delta_call(S0, K, r, sigma, T, h=0.01):
    price_up = black_scholes_call(S0 + h, K, r, sigma, T)
    price_down = black_scholes_call(S0 - h, K, r, sigma, T)
    delta = (price_up - price_down) / (2 * h)
    return delta

def bs_vega_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    return vega

def bs_gamma_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    return gamma

def bs_theta_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = (-S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    return theta

def bs_rho_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    return rho

def implied_vol_call(market_price, S0, K, r, T):
    def objective(sigma):
        return black_scholes_call(S0, K, r, sigma, T) - market_price
    return brentq(objective, 0.001, 5.0) # Search for sigma in a reasonable range
   