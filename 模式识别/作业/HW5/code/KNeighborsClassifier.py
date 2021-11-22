from numpy.core.fromnumeric import shape, size
import numpy as np

class KNeighborsClassifier():
    
    def __init__(self,n_neighbors=1):
        assert n_neighbors >=1 
        self.n_neighbors = round(n_neighbors)
        self.x_train = None
        self.y_train = None
        
    def fit(self, x_train, y_train):
        
        n1, self.d = x_train.shape
        n2 = y_train.shape[0]
        assert n1 == n2
        self.n = n1
        
        self.x_train = x_train.astype(np.float_)
        self.y_train = y_train
        
    def _predict(self, x_predict):
        d = x_predict.shape[0]
        assert d == self.d
        distances = np.zeros(self.n)
        
        for i in range(self.n):
            distances[i] = np.linalg.norm(x_predict - self.x_train[i,:])
        
        index = np.argsort(distances)
        k_index = index[:self.n_neighbors]
        k_targets = self.y_train[k_index]
        
        counts = np.bincount(k_targets)
        target = np.argmax(counts)   
        
        return target
        
    def predict(self, x_predicts):
        n, d = x_predicts.shape
        assert d == self.d
        type = self.y_train.dtype
        targets = np.zeros(n, dtype=type)
        
        for i in range(n):
            targets[i] = self._predict(x_predicts[i,:])
            
        return targets
    
    def score(self, x_predicts, y_predicts):
        n1, d = x_predicts.shape
        n2 = y_predicts.shape[0]    
        
        assert n1 == n2
        assert d == self.d
        
        targets = self.predict(x_predicts)
        return np.sum(targets==y_predicts) / n1