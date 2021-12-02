from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np
from PCA import PCA
from KNeighborsClassifier import KNeighborsClassifier as KNN
# from LDA import LDA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class Classify():
    
    def __call__(self, dataSet, **config):
        if config['dimensionality_reduction'] in [None, ""]:
            X_train = dataSet.X_train
            X_test = dataSet.X_test
        elif config['dimensionality_reduction'] == "pca":
            pca = PCA(config['n_component'])
            pca.fit(dataSet.X_train)
            X_train = pca.transform(dataSet.X_train)
            X_test = pca.transform(dataSet.X_test)
        elif config['dimensionality_reduction'] == "lda":
            lda = LDA(n_components = config['n_component'])
            lda.fit(dataSet.X_train, dataSet.y_train)
            X_train = lda.transform(dataSet.X_train)
            X_test = lda.transform(dataSet.X_test)
        else:
            raise Exception('Undefined dimensionality reduction method')
        
        knn = KNN(n_neighbors = config['n_neighbor'])
        knn.fit(X_train, dataSet.y_train)
        return knn.score(X_test, dataSet.y_test) 
