import numpy as np
import os

# 加载数据
data_path = os.path.join("..", "data", "data.txt")
data = np.loadtxt(data_path)

train_index = np.zeros(data.shape, dtype=np.bool)
for i in range(4):
    train_index[10*i:10*i+8] = True
test_index = (1-train_index).astype(np.bool)
train_data = data[train_index].reshape(-1, 3)
test_data = data[test_index].reshape(-1, 3)

# 预处理：写成增广形式
def pre_process(data):
    n,d = data.shape
    X = np.ones((n, d))
    X[:, :2] = data[:, :2].copy()
    Y = make_one_hot(data[:, 2], num_classes = 4)
    return X.T, Y.T

# 生成one-hot标签
def make_one_hot(target, num_classes):
    n = target.shape[0]
    Y = np.zeros((n, num_classes))
    for i in range(n):
        Y[i, (target[i]-1).astype(np.int)] = 1
    return Y

# 计算广义逆矩阵
def inv(X):
    n,d = X.shape[0], X.shape[1]
    I = np.eye(n)
    temp = np.dot(X, X.T) + 0.001 * I
    temp = np.linalg.inv(temp)
    return np.dot(temp, X)

# 计算准确率
def acc_rate(Y_, Y):
    n = Y.shape[1]
    acc = 0
    for i in range(n):
        target = np.argmax(Y.T[i])
        pre = np.argmax(Y_.T[i])
        if pre == target:
            acc += 1
    return acc/n


X, Y = pre_process(train_data)
print(X.shape)
print(Y.shape)
W = np.dot(inv(X),Y.T)
Y_ = np.dot(W.T, X)
print("训练集准确率", acc_rate(Y_, Y))

test_X, test_Y = pre_process(test_data)
test_Y_ = np.dot(W.T, test_X)
print("测试集准确率", acc_rate(test_Y_, test_Y))

