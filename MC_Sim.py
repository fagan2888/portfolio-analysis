# Simple Monte Carlo simulation of stocks taking covariance into account
from pandas_datareader import DataReader
from pandas import Panel, DataFrame
from scipy.linalg import cholesky
import numpy as np
import matplotlib.pyplot as plt
from pandas import bdate_range   # business days

symbols = ['AAPL', 'AMZN', 'GOOG']
data = dict((symbol, DataReader(symbol, "google", pause=1)) for symbol in symbols)
panel = Panel(data).swapaxes('items', 'minor')
closing = panel['Close'].dropna()

rets = np.log(closing / closing.shift(1)).dropna()
# TODO: Not covariance
upper_cholesky = cholesky(rets.corr(), lower=False)

n_days = 255  # Working days in a year
dates = bdate_range(start=closing.ix[-1].name, periods=n_days)
n_assets = len(symbols)
n_sims = 100
dt = 1/n_days
# dt = 1
mu = rets.mean().values
sigma = rets.std().values*np.sqrt(n_days)

rand_values = np.random.standard_normal(size = (n_days * n_sims, n_assets))
# Use the random values to generate values that satisfy the covariance
corr_values = rand_values.dot(upper_cholesky)*sigma

prices = Panel(items=range(n_sims), minor_axis=symbols, major_axis=dates)
prices.ix[:, 0, :] = closing.ix[-1].values.repeat(n_sims).reshape(n_assets,n_sims).T # set initial values

for i in range(1,n_days):
    # prices.ix[:, i, :] = prices.ix[:, i-1, :] * (np.exp(mu*dt + np.sqrt(dt)*corr_values[i::n_days])).T
    prices.ix[:, i, :] = prices.ix[:, i-1, :] * (np.exp((mu-0.5*sigma**2)*dt + np.sqrt(dt)*corr_values[i::n_days])).T

# Plot these one by one
for i in symbols:
    prices.ix[:, :, i].plot(title=i, legend=False);
    plt.show()
