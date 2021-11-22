import numpy as np
from sklearn.decomposition import PCA as P

class PCA():
    
    # n_components:保留主成分的个数
    def __init__(self, n_components):
        self.n_components = n_components
        
    # X是初始数据，n行d列，n个d维向量，每行是一个样本
    def fit(self, X):
        n,self.d = X.shape
        assert self.n_components <= self.d
        assert self.n_components <= n

        mean = np.mean(X, axis=0)
        cov = np.cov(X-mean, rowvar=False)
        eigenvalue, featurevector = np.linalg.eig(cov)
        
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
        
        mean = np.mean(X, axis=0)
        X = X - mean
        return np.dot(X, self.W)

if __name__ == '__main__':
    
    n,d = 5,2
    n_components = 1
    data = np.random.random(size=(n,d))

    pca_std = P(n_components=n_components)
    pca_std.fit(data)
    data_pca_std = pca_std.transform(data)

    pca = PCA(n_components=n_components)
    pca.fit(data)
    data_pca = pca.transform(data)

    print(np.allclose(data_pca, data_pca_std))