# Shine Shah - Stochastic Noise Modeling

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = 0
sigma = 1     

n_samples = 10000
epsilon = np.random.normal(mu, sigma, n_samples)

x = np.linspace(-4, 4, 1000)

pdf = norm.pdf(x, mu, sigma)
cdf = norm.cdf(x, mu, sigma)

# PLOT 1
plt.figure(figsize=(8,5))

plt.hist(epsilon, bins=50, density=True, alpha=0.6, label="Simulated PDF")

plt.plot(x, pdf, 'r', linewidth=2, label="Theoretical PDF")

plt.title("PDF of Stochastic Noise ε ~ N(0,1)")
plt.xlabel("ε value")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()

plt.show()


# PLOT 2 CDF
plt.figure(figsize=(8,5)

epsilon_sorted = np.sort(epsilon)
cdf_empirical = np.arange(1, n_samples+1) / n_samples

plt.plot(epsilon_sorted, cdf_empirical, label="Simulated CDF")

plt.plot(x, cdf, 'r', linewidth=2, label="Theoretical CDF")

plt.title("CDF of Stochastic Noise ε ~ N(0,1)")
plt.xlabel("ε value")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.grid()

plt.show()