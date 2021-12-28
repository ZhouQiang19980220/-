import numpy as np
from numpy.random import randint

# 读取数据
data_path = "data.txt"
X = np.loadtxt(data_path)
n, dim = X.shape

# 聚类数目
k = 5
# 随机初始化聚类中心
mus_index = np.random.randint(low=0, high=n, size=k)
mus = X[mus_index]

# 初始化标签
labels = np.zeros(n)

# 迭代次数
s = 0
max_steps = 500
isBreak = False
while s < max_steps and not isBreak:
    # 遍历每个点，并且更新聚类中心
    for i in range(n):
        distances = np.zeros(k)
        for j in range(k):
            distances[j] = np.linalg.norm(X[i] - mus[j])
        labels[i] = np.argmin(distances)  

    # 更新聚类中心
    error = 0
    for j in range(k):
        x = X[labels == j]
        temp = np.mean(x, axis=0)
        error += np.linalg.norm(temp - mus[j])
        mus[j] = temp
    # 计算误差
    isBreak = (error < 1e-3)
    s += 1
print("s = ", s)
np.set_printoptions(precision=3)
print(mus)
