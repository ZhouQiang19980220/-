import os
from tqdm import tqdm
from classify import Classify 
import matplotlib.pyplot as plt 

# # 导入数据:每行是一个样本
data_path = os.path.join("data", "ORLData_25.mat")
classify = Classify(data_path=data_path)

# 设置超参数
# config = dict()
# config['n_component'] = 10
# config['n_neighbor'] = 1
# config['dimensionality_reduction'] = "pca"


# n_components = [i for i in range(5, 320, 5)]
# scores = []
# for n_component in tqdm(n_components):
#     config['n_component'] = n_component
#     scores.append(classify(**config))
    
config = dict()
config['n_component'] = 39
config['n_neighbor'] = 1
config['dimensionality_reduction'] = "lda"
n_components = [i for i in range(50, 5, -5)]

scores = []
for n_component in tqdm(n_components):
    config['n_component'] = n_component
    scores.append(classify(**config))
    
plt.plot(n_components ,scores)
plt.show()
    
