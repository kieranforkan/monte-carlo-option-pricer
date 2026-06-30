import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm(S0, mu, sigma, T, N):
    dt = T / N
    S = np.zeros(N+1)         # Array to hold the path of the stock price
    S[0] = S0                 # Starting value of the stock price
    for n in range(N):
        Z = np.random.normal()       # One standard normal draw
        S[n+1] = S[n] + mu * S[n] * dt + sigma * S[n] * np.sqrt(dt) * Z  # GBM formula
    return S

M = 1000
all_paths = []

for i in range(M):
    path = simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1, N=252)
    all_paths.append(path)
    plt.plot(path, color = "steelblue", alpha=0.03)

all_paths = np.array(all_paths)
mean_path = np.mean(all_paths, axis=0)

plt.plot(mean_path, color = "darkblue", label="Mean Path", linewidth=2)
final_price = mean_path[-1]
plt.axhline(y=final_price, color='red', linewidth=1.35, linestyle='--', label=f"Final Mean Price: {final_price:.2f}")
plt.legend()
plt.xlabel("Time Step")
plt.ylabel("Stock Price")
plt.title(f"{M} Simulated GBM Paths")
plt.show()