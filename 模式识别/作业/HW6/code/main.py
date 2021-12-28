import numpy as np
from kMeans import kMeans
import matplotlib.pyplot as plt
from acc import rand_index_score

# 真实误差
mus = np.array([
    [1, -1],
    [5.5, -4.5],
    [1, 4],
    [6, 4.5],
    [9, 0.0]
    ])

num = 200
labels = np.zeros(num * mus.shape[0], dtype=np.uint8)
for i in range(mus.shape[0]):
    labels[i*num:(i+1)*num] = i

# 读取数据
data_path = "data.txt"
X = np.loadtxt(data_path)
k = 5
mus_pred, labels_pred = kMeans(X, k)

# 绘图
for i, mu in enumerate(mus_pred):
    x = X[labels_pred == i, :]
    color = plt.cm.Set1(i)
    plt.scatter(x[:,0], x[:,1], color = color, s = 10)
    color = plt.cm.Set3(i)
    plt.scatter(mu[0], mu[1], s=500, color = color, marker = '*')
plt.show()

# 计算精度
print("聚类精度:", rand_index_score(labels_pred, labels))
# 计算聚类中心距离
mus = np.sort(mus, axis=0)
mus_pred = np.sort(mus_pred, axis=0)
dis = 0.0
for i in range(mus.shape[0]):
    dis += np.linalg.norm(mus_pred[i] - mus[i])
print("预测中心与真实中心的距离:", dis)