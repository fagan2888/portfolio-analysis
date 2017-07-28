# Portfolio volatility prediction using Principal Component Analysis
from pandas_datareader import DataReader
from pandas import Panel, DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sk_pca

n_days = 1000  # Return series length

n_assets = 5
# Observed that 2 principal components contributed to  >90% variance
# most of the time
n_components = 2
n_portfolios = 100
closing = np.random.randint(1300, 1500, n_assets)/100 # Closing price between 13 and 15
dt = 1/n_days
mu = np.random.randint(n_assets, 15, n_assets)/10000  # Mean between .0005 and .0015 (Daily)
print(mu)

sigma = np.random.randint(1, 20, n_assets)/100  # Random volatility between 1-20% (Annual)
print(sigma)
# sigma = np.zeros(n_assets)
# NOTE: Performance wise, it's better to switch the dimensions while
# asssigning, so as to avoid getting transpose for calculating covariance
# This is not done for understandability right now
prices = np.full((n_days, n_assets), closing)

# TODO: Document assumptions
daily_returns = np.random.standard_normal(size = (n_days , n_assets))
# Limit the random returns
daily_returns = daily_returns/daily_returns.sum()

# Simulate stocks. This is just to simulate the variance.
for i in range(1, n_days):
    prices[i, :] = prices[i-1, :] * (np.exp((mu-0.5*sigma**2)*dt +
        np.sqrt(dt)*daily_returns[i])).T

cov_matrix = np.cov(prices.T)
plt.plot(prices)
plt.show()

weight_list = []
results = np.zeros((2, n_portfolios))
# Calculate portfolio volatility
for i in range(n_portfolios):
    randarr = np.random.rand(n_assets)
    weights = randarr/randarr.sum()  # Five weights summing to 1
    weight_list.append(weights)
    # calculate annualised portfolio return
    pf_ret = round(np.sum(mu * weights) * n_days, 2)
    pf_volatility = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(n_days)
    results[0,i] = pf_ret
    results[1,i] = pf_volatility

pca = sk_pca(n_components=n_assets)
pc = pca.fit_transform(prices)
# plot the variance explained by pcs
# plt.bar(range(n_assets), pca.explained_variance_ratio_)
# plt.title('variance explained by pc')
# plt.show()

# Select a portfolio out of the 100
selected_pf = 1

# get First `n_components` Principal components
pcs_1_2 = pca.components_[0:n_components,:].T
pc_weights = weight_list[selected_pf].dot(pcs_1_2)

# pf_volatility = round(np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(n_days),2)
# Portfolio volatility with PCs
pf_volatility = np.sqrt(np.sum([ pc_weights[i]**2 * np.var(prices[:,i]) for i in range(n_components)
    ])) * np.sqrt(n_days)

print("Actual volatility: " , results[1, selected_pf])
print("Volatility calculated by PCA : " , pf_volatility)
