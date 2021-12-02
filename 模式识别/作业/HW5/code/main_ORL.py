import os
from tqdm import tqdm
from classify import Classify 
import matplotlib.pyplot as plt
from dataSet import DataSet

if __name__ == '__main__':    
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    
    config = dict()
    config['n_neighbor'] = 1
    config['dimensionality_reduction'] = None
    config['data_path'] = os.path.join("data", "ORLData_25.mat")
    config['data_type'] = "ORL"
    # # 导入数据:每行是一个样本
    vehicle = DataSet(config['data_path'], config['data_type'])
    classify = Classify()
    
    score = classify(vehicle, **config)

    # PCA降维
    n_components_pca = [i for i in range(1, 320, 5)]
    config['dimensionality_reduction'] = "pca"
    scores_pca = []
    for n_component in tqdm(n_components_pca):
        config['n_component'] = n_component
        scores_pca.append(classify(vehicle, **config))
        
    figure1 = plt.figure()
    plt.plot(n_components_pca ,scores_pca, label="pca降维")
    scores = [score for i in n_components_pca]
    plt.plot(n_components_pca, scores, label="不降维")
    plt.legend()
    plt.title("特征维数对分类性能的影响\n  分类方式：KNN(k=1) \n 数据集：ORL")
    plt.xlabel("特征维数")
    plt.ylabel("分类准确率")
        
    # LDA降维
    n_components_lda = [i for i in range(1, 40, 5)]
    config['dimensionality_reduction'] = "lda"
    scores_lda = []
    for n_component in tqdm(n_components_lda):
        config['n_component'] = n_component
        scores_lda.append(classify(vehicle, **config))
    
    figure2 = plt.figure()
    plt.plot(n_components_lda ,scores_lda, label="lda降维")
    scores = [score for i in n_components_lda]
    plt.plot(n_components_lda, scores, label="不降维")
    plt.legend()
    plt.title("特征维数对分类性能的影响\n  分类方式：KNN(k=1) \n 数据集：ORL")
    plt.xlabel("特征维数")
    plt.ylabel("分类准确率")
    
    # 降维方式对比
    figure3 = plt.figure()
    n = len(n_components_lda)
    plt.plot(n_components_lda ,scores_lda, label="lda降维")
    plt.plot(n_components_lda ,scores_pca[:n], label="pca降维")
    plt.plot(n_components_lda ,scores[:n], label="不降维")
    plt.legend()
    plt.title("降维方式对分类性能的影响\n  分类方式：KNN(k=1) \n 数据集：ORL")
    plt.xlabel("特征维数")
    plt.ylabel("分类准确率")

    plt.show()