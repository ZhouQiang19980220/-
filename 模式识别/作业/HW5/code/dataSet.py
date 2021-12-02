from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np

# 载入数据
class DataSet():
    def __init__(self, data_path, data_type="ORL"):
        if data_type == "ORL":
            ORL = loadmat(data_path)
            data = ORL["ORLData"].T
        elif data_type == "vehicle":
            data = loadmat(data_path)
            data = data['UCI_entropy_data']['train_data'][0][0].T
        else:
            raise Exception('Undefined data type!')
        
        target = data[:,-1]
        x = data[:,:-1].astype(np.float_)
        self.n, self.d = data.shape

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, target, test_size = 0.2, random_state = 0)