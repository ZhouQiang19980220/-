#%%
# 生成数据
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
sigma = np.eye(2)
mus = np.array([
    [1, -1],
    [5.5, -4.5],
    [1, 4],
    [6, 4.5],
    [9, 0.0]
    ])
num = 200
X = np.zeros([num * mus.shape[0], sigma.shape[0]])
for i, mu in enumerate(mus):
    x = np.random.multivariate_normal(mu, sigma, num)
    plt.scatter(x[:,0], x[:,1])
    X[i*num:(i+1)*num] = x

data_path = "data.txt"
np.savetxt(data_path, X)
plt.show()