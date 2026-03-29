#PDF and CDF of Motion Perturbation (Laplace)
from scipy.stats import laplace
import matplotlib.pyplot as plt
import numpy as np
# Laplace parameter
b = 1.0  # scale

x = np.linspace(-6, 6, 1000)

pdf_l = laplace.pdf(x, loc=0, scale=b)
cdf_l = laplace.cdf(x, loc=0, scale=b)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(x, pdf_l)
plt.title("PDF of Motion Perturbation (Laplace)")
plt.xlabel("Perturbation")
plt.ylabel("PDF")

plt.subplot(1,2,2)
plt.plot(x, cdf_l)
plt.title("CDF of Motion Perturbation (Laplace)")
plt.xlabel("Perturbation")
plt.ylabel("CDF")

plt.tight_layout()
plt.show()

#Laplace Motion Perturbations
import numpy as np
import matplotlib.pyplot as plt

# Parameters
b = 2.0        # Laplace scale (controls tail heaviness)
N = 10000       # number of time steps

# Motion perturbations
eps_par  = np.random.laplace(loc=0, scale=b, size=N)
eps_perp = np.random.laplace(loc=0, scale=b, size=N)

# Plot distributions
plt.figure(figsize=(8,5))
plt.hist(eps_par, bins=200, density=True, alpha=0.6, label="Parallel")
plt.hist(eps_perp, bins=200, density=True, alpha=0.6, label="Perpendicular")

plt.xlabel("Perturbation Value")
plt.ylabel("Probability Density")
plt.title("Laplace Motion Perturbations")
plt.legend()
plt.show()

#Correlated Motion Perturbations
rho = 0.8  # temporal correlation
eps_par = np.zeros(N)
eps_perp = np.zeros(N)

for t in range(1, N):
    eps_par[t]  = rho*eps_par[t-1]  + np.random.laplace(0, b)
    eps_perp[t] = rho*eps_perp[t-1] + np.random.laplace(0, b)

plt.figure(figsize=(8,4))
plt.plot(eps_par[:500], label="Parallel")
plt.plot(eps_perp[:500], label="Perpendicular")

plt.xlabel("Time Step")                 
plt.ylabel("Motion Perturbation")       

plt.legend()
plt.title("Correlated Motion Perturbations")
plt.show()

#Heavy Tail Motion Uncertainty (Laplace vs Gaussian comparison)
import numpy as np
import matplotlib.pyplot as plt
eps_gauss = np.random.normal(0, np.sqrt(2)*b, N)
plt.figure(figsize=(8,5))
plt.hist(eps_par, bins=200, density=True, alpha=0.6, label="Laplace")
plt.hist(eps_gauss, bins=200, density=True, alpha=0.6, label="Gaussian")

plt.xlim(-20, 20)
plt.xlabel("Motion Perturbation Value")     
plt.ylabel("Probability Density")         

plt.legend()
plt.title("Heavy Tail Motion Uncertainty (Laplace vs Gaussian)")
plt.show()

