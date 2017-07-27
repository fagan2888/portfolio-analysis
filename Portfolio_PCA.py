from pandas_datareader import DataReader
from pandas import Panel, DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sk_pca

n_days = 255  # Working days in a year

n_assets = 5
# Observed that 2 principal components contributed to  >90% variance
# most of the time
n_components = 2
n_portfolios = 100
closing = np.random.randint(150, 1500, n_assets)/100
dt = 1/n_days
mu = np.random.randint(n_assets, 15, n_assets)/10000  # Mean between .0005 and .0015 (Daily)

sigma = np.random.randint(10, 20, n_assets)/100  # Random volatility between 10-20% (Annual)
prices = np.full((n_days, n_assets), closing)  # set initial values

# TODO: Document assumptions
daily_returns = np.random.standard_normal(size = (n_days , n_assets))
cov_matrix = np.cov(daily_returns.T.shape)

# Simulate stocks. This is just to get the covariance.
for i in range(1 ,n_days):
    prices[i, :] = prices[i-1, :] * (np.exp((mu-0.5*sigma**2)*dt +
        np.sqrt(dt)*daily_returns[i])).T

# plt.plot(prices)
# plt.show()
weight_list = []
results = np.zeros((2, n_portfolios))
for i in range(n_portfolios):
    randarr = np.random.rand(n_assets)
    weights = randarr/randarr.sum()  # Five weights summing to 1
    weight_list.append(weights)
    #calculate annualised portfolio return
    portfolio_return = round(np.sum(mu * weights) * 252,2)
    portfolio_volatility = round(np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252),2)
    results[0,i] = portfolio_return
    results[1,i] = portfolio_volatility

results_frame = DataFrame(results.T,columns=['ret','stdev'])
# plt.scatter(results_frame.stdev,results_frame.ret,cmap='RdYlBu')
# plt.show()

pca = sk_pca(n_components=n_components)  # Take the first 5 components
# rets = np.log(closing / closing.shift(1)).dropna()
pc = pca.fit_transform(prices)

# plot the variance explained by pcs
plt.bar(range(n_components), pca.explained_variance_ratio_)
plt.title('variance explained by pc')
plt.show()

# print(pca.explained_variance_ratio_)

# get the Principal components
pcs = pca.components_
weights = weight_list[1]
# first component
pc1 = pcs[0,:]
# normalized to 1
print(pc1)
print(np.asmatrix(pc1/sum(pc1)))
pc_w = np.asmatrix(pc1/sum(pc1)).T

pf_ret = mu * weights * n_days
# apply our first componenet as weight of the stocks
pc1_ret = pf_ret*pc_w

# plot the total return index of the first PC portfolio
pc_ret = DataFrame(data=pc1_ret)
pc_ret_idx = pc_ret+1

# Pad the explained_variance_ with zeros, multiply it with the weight vector
portfolio_volatility_pca = weights * np.pad(np.sqrt(pca.explained_variance_), (0,n_assets-n_components), mode='constant')

print("PCA portfolio_volatility: ", sum(portfolio_volatility_pca))
print("Actual " , results[1,1])

# pc_ret_idx= pc_ret_idx.cumprod()
# pc_ret_idx.columns = ['pc1']
#
# print(prices)
# print(weights)
#
# pc_ret_idx['actual'] = mu * weights
# pc_ret_idx.plot(subplots=True,title ='PC portfolio vs Market',layout =[1,2])
#
# # plot the weights in the PC
# weights_df = DataFrame(data = pc_w*100,index = stocks_.columns)
# weights_df.columns=['weights']
# weights_df.plot.bar(title='PCA portfolio weights',rot = 45,fontsize = 8)
