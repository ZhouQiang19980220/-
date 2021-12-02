import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_std
import matplotlib.pyplot as plt

class LDA():
    
    # n_components:降到多少维
    def __init__(self, n_components):
        self.n_components = n_components
    
    # n个d维数据，n*d矩阵，每一行是一个数据
    def fit(self, X_train, y_train):
        # 保证样本数相等
        n1, self.d = X_train.shape
        n2 = y_train.shape[0]
        assert n1 == n2
        self.n = n1
        
        clusters = np.unique(y_train)   
        assert self.n_components < self.d        
        assert self.n_components < clusters.shape[0]
        
        # 全局均值
        mean = np.mean(X_train, axis=0)            
        # 类内散度矩阵
        Sw = np.zeros(shape=(self.d, self.d), dtype=np.float_)
        # 类间散度矩阵
        Sb = np.zeros(shape=Sw.shape, dtype = np.float_)
        for cluster in clusters:
            # 第i类数据
            X_i = X_train[y_train==cluster]
            N_i = X_i.shape[0]
            mean_i = np.mean(X_i, axis=0)
            X_i = X_i - mean_i
            cov_i = np.cov(X_i, rowvar=False)
            Sw += cov_i
            # 将向量转成矩阵
            temp = (mean_i - mean).reshape(1, -1)
            Sb += N_i * np.dot(temp.T, temp)
        S =  np.dot(np.linalg.inv(Sw + 10 ** (-3) * np.eye(self.d)), Sb)
        # 取S的前k大的特征值对应的特征向量，组成变换矩阵       
        eigenvalue, featurevector = np.linalg.eig(S)
        
        # 将特征值升序排列，index是原数组中的下标
        index = np.argsort(eigenvalue)
        # 取最大的n个特征值对应的下标
        n_index = index[-self.n_components:]
        # 取对应的列(W是d行m列)
        self.W = featurevector[:,n_index].real
    
    def transform(self, X):
        n,d = X.shape
        # 数据维度必须一样
        assert d == self.d
        
        return np.dot(X, self.W)
    
if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target
    lda = LDA(n_components = 2)
    lda.fit(X, Y)
    
    data_1 = lda.transform(X)
    data_2 = LDA_std(n_components=2).fit_transform(X, Y)


    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("LDA")
    plt.scatter(data_1[:, 1], data_1[:, 0], c = Y)

    plt.subplot(122)
    plt.title("sklearn_LDA")
    plt.scatter(data_2[:, 1], data_2[:, 0], c = Y)
    plt.savefig("LDA.png",dpi=600)
    plt.show()
