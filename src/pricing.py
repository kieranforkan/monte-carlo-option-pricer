import numpy as np
from src.simulation import simulate_gbm

def price_european_call(S0, K, r, sigma, T, N, M):
    payoffs = np.zeros(M)          # One payoff per simulated path
    for i in range (M):
        path = simulate_gbm(S0, r, sigma, T, N)        # drift = r
        S_T = path[-1]
        payoffs[i] = np.maximum(S_T - K, 0)      # Call payoff
    discounted = np.exp(-r * T) * np.mean(payoffs)    # Average payoff
    return discounted