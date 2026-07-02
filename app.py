import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.pricing import (
    black_scholes_call, price_european_call,
    bs_delta_call, bs_vega_call, bs_gamma_call, bs_theta_call, bs_rho_call,
)

st.title("Monte Carlo Option Pricer")
st.write("Interactive European call option pricing and Greeks.")

# --- Sidebar inputs (the sliders) ---
st.sidebar.header("Parameters")
S0    = st.sidebar.slider("Stock price (S)", 50.0, 150.0, 100.0)
K     = st.sidebar.slider("Strike (K)", 50.0, 150.0, 100.0)
r     = st.sidebar.slider("Risk-free rate (r)", 0.0, 0.15, 0.05)
sigma = st.sidebar.slider("Volatility (σ)", 0.05, 0.60, 0.20)
T     = st.sidebar.slider("Time to expiry (years)", 0.1, 3.0, 1.0)

# --- Prices ---
bs_price = black_scholes_call(S0, K, r, sigma, T)
mc_price = price_european_call(S0, K, r, sigma, T, N=252, M=10000)

col1, col2 = st.columns(2)
col1.metric("Black–Scholes price", f"{bs_price:.4f}")
col2.metric("Monte Carlo price", f"{mc_price:.4f}")

# --- Greeks ---
st.subheader("Greeks")
g1, g2, g3, g4, g5 = st.columns(5)
g1.metric("Delta", f"{bs_delta_call(S0, K, r, sigma, T):.4f}")
g2.metric("Vega",  f"{bs_vega_call(S0, K, r, sigma, T):.4f}")
g3.metric("Gamma", f"{bs_gamma_call(S0, K, r, sigma, T):.4f}")
g4.metric("Theta", f"{bs_theta_call(S0, K, r, sigma, T):.4f}")
g5.metric("Rho",   f"{bs_rho_call(S0, K, r, sigma, T):.4f}")

# --- Payoff diagram ---
st.subheader("Payoff at expiry")
prices = np.linspace(50, 150, 100)
payoff = np.maximum(prices - K, 0)
fig, ax = plt.subplots()
ax.plot(prices, payoff, color="steelblue")
ax.axvline(K, color="red", linestyle="--", alpha=0.5, label="Strike")
ax.set_xlabel("Stock price at expiry"); ax.set_ylabel("Payoff"); ax.legend()
st.pyplot(fig)