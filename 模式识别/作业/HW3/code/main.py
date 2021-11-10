import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size

# 针对二分类问题进行规范化增广样本
def pre_process(data, classes):
    res = []
    m,n = data.shape
    for i in range(m):
        if data[i][2] in classes:
            temp = data[i].copy()
            if temp[2] == classes[0]:
                temp[2] = 1
            else:
                temp = -temp
                temp[2] = -1
            res.append(temp)
    return np.array(res)

def draw(classes, a):
    target = data[:,2]
    # 绘图
    for c in classes:
        index = (target == c)
        plt.plot(data[index, 0], data[index, 1],'.', label='sample of class ' + str(c))

    x = np.arange(-10, 10, 0.01)
    y = -(a[0] / a[1]) * x - a[2] / a[1]
    plt.plot(x, y, '-', label='boundary of class' + str(classes[0]) + " and class " + str(classes[1]))
    plt.legend()
    plt.show()


def batch_perception(data, classes, max_step=100):
    data = pre_process(data, classes)
    # 初始化权重向量
    a = np.zeros(3)
    step = 0
    while 1:
        # 错误样本集合
        mistakes = []
        for y in data:
            if np.dot(a, y) <= 0:
                mistakes.append(y)
        step += 1
        # 如果错误样本集为空，则全部分类正确，迭代结束，退出循环
        if mistakes == [] or step >= max_step:
            break
        else:
            mistakes = np.array(mistakes)
            k = np.sum(mistakes, axis=0)
            a = a + np.sum(mistakes, axis=0)
    print("step = ", step)
    draw(classes, a)
    return a

# 计算广义逆矩阵
def inv(Y):
    n,d = Y.shape[0], Y.shape[1]
    I = np.eye(d)
    temp = np.dot(Y.T, Y) + 0.001 * I
    temp = np.linalg.inv(temp)
    return np.dot(temp, Y.T)

def ho_Kashyap(data, classes, max_step=100):
    data = pre_process(data, classes)
    n = data.shape[0]
    # 初始化
    b_min = 0.001
    b = np.zeros(n) + 0.1
    data_inv = inv(data)
    a = np.dot(data_inv, b)
    k = 1
    e = np.dot(data, a) - b
    while True:
        e_ = e + np.abs(e)
        # 计算误差
        b = b + (1/k) * e_
        a = np.dot(data_inv, b)
        e = np.dot(data, a) - b
        k = k + 1

        # 误差足够小，则可以结束迭代
        if np.linalg.norm(e) < b_min or k >= max_step:
            break
    # 绘制示意图
    draw(classes, a)
    # 计算错误率
    e_r = error_rate(data, a)
    print("class " + str(classes[0]) + "和class " + str(classes[1]) + "的错误率:", e_r)
    return a,b

# 这里的参数时规范化增广样本，a是齐次权重向量
def error_rate(data, a):
    n = data.shape[0]
    error = 0
    for i in range(n):
        if np.dot(data[i], a) < 0:
            error += 1
    return error / n

# 加载数据
data_path = os.path.join("..", "data", "data.txt")
data = np.loadtxt(data_path)
# # 计算权重向量
# a = batch_perception(data = data, classes = [1, 2])
# a = batch_perception(data = data, classes = [2, 3])
a,b = ho_Kashyap(data, [1,3])
a,b = ho_Kashyap(data, [2,4])