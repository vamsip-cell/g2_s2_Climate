# Chebyshev Rainfall Analysis - Complete Script

# PDF and CDF of Monthly Average Rainfall (Chebyshev Context)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = 50
sigma = 10
n = 30

mean_sample = mu
std_sample = sigma / np.sqrt(n)

x = np.linspace(30, 70, 500)

pdf_values = norm.pdf(x, mean_sample, std_sample)
cdf_values = norm.cdf(x, mean_sample, std_sample)

fig, axes = plt.subplots(1, 2, figsize=(14,6))

axes[0].plot(x, pdf_values)
axes[0].set_xlabel("Monthly Average Rainfall (mm)")
axes[0].set_ylabel("Probability Density")
axes[0].set_title("PDF of Monthly Average Rainfall (Chebyshev)")
axes[0].grid(True)

axes[1].plot(x, cdf_values)
axes[1].set_xlabel("Monthly Average Rainfall (mm)")
axes[1].set_ylabel("Cumulative Probability")
axes[1].set_title("CDF of Monthly Average Rainfall (Chebyshev)")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Chebyshev Bound vs Actual Gaussian Tail Probability
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

sigma = 10
n = 30

sigma_mean = sigma / np.sqrt(n)

deviation = np.linspace(1, 30, 300)

chebyshev_bound = sigma_mean**2 / (deviation**2)
actual_probability = 2 * (1 - norm.cdf(deviation / sigma_mean))

plt.figure(figsize=(8,6))
plt.plot(deviation, chebyshev_bound)
plt.plot(deviation, actual_probability)
plt.xlabel("Deviation from Mean Rainfall (mm)", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.title("Chebyshev Bound vs Actual Gaussian Probability", fontsize=14)
plt.legend(["Chebyshev Bound", "Gaussian Tail Probability"])
plt.grid(True)
plt.tight_layout()
plt.show()

# Effect of Sample Size on Chebyshev Bound
import numpy as np
import matplotlib.pyplot as plt

sigma = 10
fixed_deviation = 10

n_values = np.linspace(5, 200, 300)

bound_vs_n = sigma**2 / (n_values * fixed_deviation**2)

plt.figure(figsize=(8,6))
plt.plot(n_values, bound_vs_n)
plt.xlabel("Number of Observed Days (n)", fontsize=12)
plt.ylabel("Chebyshev Upper Bound", fontsize=12)
plt.title("Effect of Sample Size on Rainfall Deviation Bound", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Chebyshev Inequality Bound vs Deviation
import numpy as np
import matplotlib.pyplot as plt

sigma = 10
n = 30

sigma_mean = sigma / np.sqrt(n)

deviation = np.linspace(1, 30, 300)

chebyshev_bound = sigma_mean**2 / (deviation**2)

plt.figure(figsize=(8,6))
plt.plot(deviation, chebyshev_bound)
plt.xlabel("Deviation from Mean Rainfall (mm)", fontsize=12)
plt.ylabel("Chebyshev Upper Bound", fontsize=12)
plt.title("Chebyshev Inequality: Bound vs Rainfall Deviation", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()
