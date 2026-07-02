import numpy as np
from src.simulation import simulate_gbm
from scipy.stats import norm

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