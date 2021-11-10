import os
import struct
import numpy as np
from sklearn.decomposition import PCA
import logging

# 加载数据集
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
 
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    return select_data(images, labels)

# 筛选数据集：根据class_num
def select_data(x,y):
    x = x[y<class_num]
    y = y[y<class_num]
    return x, y

# 预处理：PCA降维
def preProcess(X_train, X_test, d=100):
    decomposition = PCA(n_components=d)
    X_train = decomposition.fit_transform(X_train)
    X_test = decomposition.transform(X_test)

    return X_train, X_test

# 训练：计算先验概率和各类的均值以及协方差矩阵
def train(X_train, Y_train):
    l,d = X_train.shape;
    # 先验概率
    P = np.zeros(class_num)
    # 均值
    means = np.zeros([class_num,d])
    # 协方差矩阵
    covs = np.zeros([class_num,d,d])

    for i in range(class_num):
        x = X_train[Y_train == i]
        P[i] = x.shape[0] / l
        means[i] = np.mean(x, axis=0)
        covs[i] = np.cov(x, rowvar=False) #+ 0.01 * np.eye(d)

    return P, means, covs


# LDF：不同的类共用一个协方差矩阵
# 参数含义 均值 协方差的逆 先验概率 测试集
def LDF(means, covs, P, X_test):
    # 特征向量维数
    d = means.shape[1]
    # 计算协方差矩阵：各类协方差矩阵的加权和
    cov = np.zeros([d,d])
    for i in range(class_num):
        cov = cov + P[i] * covs[i]
    cov_inv = np.linalg.inv(cov)
    # 斜率
    w1 = np.zeros([class_num,d])
    # 偏置
    w0 = np.zeros(class_num)
    res = np.zeros([class_num, X_test.shape[0]])

    # 计算线性判别函数
    for i in range(class_num):
        w1[i] = np.dot(cov_inv, means[i])
        # print(w1[i])
        w0[i] = -0.5 * np.dot(np.dot(means[i].T,cov_inv),means[i]) + np.log(P[i])
        # print(w0[i])

        for idx, x in enumerate(X_test):
            res[i][idx] = np.dot(w1[i], x) + w0[i]

    return np.argmax(res, axis=0)

# QDF：一般情况
# 参数含义 均值 协方差的逆 先验概率 测试集
def QDF(means, covs, P, X_test):
    # 特征向量维数
    d = means.shape[1]
    # 二次项系数
    w2 = np.zeros([class_num,d,d])
    # 一次项系数
    w1 = np.zeros([class_num,d])
    # 常数项
    w0 = np.zeros([class_num])

    res = np.zeros([class_num, X_test.shape[0]])


    # 计算二次判别函数
    for i in range(class_num):
        w2[i] = -0.5 * np.linalg.inv(covs[i])
        w1[i] = -2 * np.dot(w2[i], means[i])
        w0[i] = np.dot(np.dot(means[i].T,w2[i]),means[i]) + np.log(P[i]) - 0.5 * np.log(np.linalg.det(covs[i]))
        for idx, x in enumerate(X_test):
            res[i][idx] = np.dot(np.dot(x.T, w2[i]),x) + np.dot(w1[i], x) + w0[i]

    return np.argmax(res, axis=0)

# 综合LDF和QDF
def test(means, covs, P, X_test,method="LDF"):
    if method == "LDF":
        return LDF(means, covs, P, X_test)
    elif method == "QDF":
        return QDF(means, covs, P, X_test)
    else:
        print("Method not supported")

def main():
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    file_name = method + "_" + str(d) + "_" + str(class_num) + ".txt"
    log_file = os.path.join("..", "log", file_name)
    logging.basicConfig(filename=log_file, level=logging.INFO, format=LOG_FORMAT)
    logging.info('method: %s', method)
    logging.info('decomposition_num: %s' % d)
    logging.info('class_num: %d', class_num)
    print('method: ', method)
    print('decomposition_num: ', d)
    print('class_num: ', class_num)
    # 加载数据集
    X_train, Y_train = load_mnist('../data/') 
    X_test, Y_test = load_mnist('../data/', 't10k')

    # 预处理：PCA降维
    X_train, X_test = preProcess(X_train, X_test, d)

    # 训练
    P, means, covs = train(X_train, Y_train)

    # LDF
    res = test(means, covs, P, X_test,method)
    acc = np.sum(res == Y_test) / Y_test.shape
    logging.info('accuracy: %s', acc[0])
    print('accuracy: ', acc[0])

    
    
# 训练
if __name__ == '__main__':

    # 类别数目
    class_num = 10
    # 方法
    method = "QDF"
    # 降维数目
    d =80
    main()