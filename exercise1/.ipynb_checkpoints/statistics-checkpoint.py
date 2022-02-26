### Statistics
#Have a look at the ```scipy.stats``` [module](https://docs.scipy.org/doc/scipy/reference/stats.html)

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

#### a. Create a discrete random variable with poissonian distribution and plot its probability mass function (PMF), cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable

# Generate discrete variable
pvar = stats.poisson
mu = 1.0
xf = 20
x = np.linspace(0, xf, xf+1)

d_plot_kwargs = dict(linestyle='none', marker='o', c='r')

# Plot PMF
plt.plot(x, pvar.pmf(x, mu), **d_plot_kwargs)
plt.title('Poisson Variable - PMF')
plt.savefig('poisson-PMF.pdf')
plt.clf()

# Plot CDF
plt.plot(x, pvar.cdf(x, mu), **d_plot_kwargs)
plt.title('Poisson Variable - CDF')
plt.savefig('poisson-CDF.pdf')
plt.clf()

# Plot histogram
hist=pvar.rvs(mu, size=1000)
binwidth = 1
plt.hist(hist, bins=range(min(hist), max(hist) + binwidth, binwidth))
plt.title('Poisson Variable - HIST')
plt.savefig('poisson-HIST.pdf')
plt.clf()

#### b. Create a continious random variable with normal distribution and plot its probability mass function (PMF), cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable

# Generate continuous variable
cvar = stats.norm
xf = 5
x = np.linspace(-xf, xf)

c_plot_kwargs = dict(lw=2, marker='o', c='r')

# PLot PDF
plt.plot(x, cvar.pdf(x), **c_plot_kwargs)
plt.title('Normal Variable - PDF')
plt.savefig('normal-PDF.pdf')
plt.clf()

# PLot CDF
plt.plot(x, cvar.cdf(x), **c_plot_kwargs)
plt.title('Normal Variable - CDF')
plt.savefig('normal-CDF.pdf')
plt.clf()

# Plot histogram
hist=cvar.rvs(size=1000)
plt.hist(hist)
plt.title('Normal Variable - HIST')
plt.savefig('normal-HIST.pdf')
plt.clf()

#### c. Test if two sets of (independent) random data comes from the same distribution
#Hint: Have a look at the ```ttest_ind``` function

# Generate two independent variables from same distribution
rvs1 = stats.norm.rvs(size=500)
rvs2 = stats.norm.rvs(size=500)
print("Testing two normal distributions:")
result = stats.ttest_ind(rvs1, rvs2)
print("statistic: ", result.statistic)
print("pvalue: ", result.pvalue)

# Generate two independent variables from different distributions
rvs1 = stats.poisson.rvs(1, size=500)
rvs2 = stats.norm.rvs(size=500)
print("\nComparing a normal distribution with a Poisson:")
result = stats.ttest_ind(rvs1, rvs2)
print("statistic: ", result.statistic)
print("pvalue: ", result.pvalue)