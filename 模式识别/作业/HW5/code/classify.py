from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np
from PCA import PCA
from KNeighborsClassifier import KNeighborsClassifier as KNN
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from LDA import LDA
import matplotlib.pyplot as plt

class Classify():
    
    def __init__(self, data_path):
        ORL = loadmat(data_path)
        x = ORL["ORLData"].T
        target = x[:,-1]
        x = x[:,:-1].astype(np.float_)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, target, test_size = 0.2, random_state = 0)
        
    def __call__(self, **config):
        if config['dimensionality_reduction'] == "pca":
            pca = PCA(config['n_component'])
            pca.fit(self.X_train)
            X_train = pca.transform(self.X_train)
            X_test = pca.transform(self.X_test)
        elif config['dimensionality_reduction'] == "lda":
            lda = LDA(n_components = config['n_component'])
            lda.fit(self.X_train, self.y_train)
            X_train = lda.transform(self.X_train)
            X_test = lda.transform(self.X_test)
        elif config['dimensionality_reduction'] in [None, ""]:
            X_train = self.X_train
            X_test = self.X_test
        else:
            raise Exception('Undefined dimensionality reduction method')
        
        knn = KNN(n_neighbors = config['n_neighbor'])
        knn.fit(X_train, self.y_train)
        return knn.score(X_test, self.y_test) 
