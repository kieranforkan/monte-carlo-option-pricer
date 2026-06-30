import numpy as np

def simulate_gbm(S0, mu, sigma, T, N):
    dt = T / N
    S = np.zeros(N+1)         # Array to hold the path of the stock price
    S[0] = S0                 # Starting value of the stock price
    for n in range(N):
        Z = np.random.normal()       # One standard normal draw
        S[n+1] = S[n] + mu * S[n] * dt + sigma * S[n] * np.sqrt(dt) * Z  # GBM formula
    return S

def simulate_ou(X0, theta, mu, sigma, T, N):
    dt = T / N
    X = np.zeros(N+1)         # Array to hold the path of the process
    X[0] = X0                 # Starting value of the process
    for n in range(N):
        Z = np.random.normal()       # One standard normal draw
        X[n+1] = X[n] + theta * (mu - X[n]) * dt + sigma * np.sqrt(dt) * Z  # OU formula
    return X
