import os
from torch.utils.data import DataLoader
from config import config
from MyDataSet import MyDataSet
from dnn import Net
from train import train
from utils import draw
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 加载数据集
    data_path = os.path.join("data", "data.txt")
    target_path = os.path.join("data", "target.txt")
    trainSet = MyDataSet(data_path, target_path)
    trainLoader = DataLoader(trainSet, batch_size = config['batch_size'])
    
    # 训练并测试
    net = Net(3, config['hidden_channels'], 3)
    acc_record = train(trainLoader, config, net)

    # 计算准确率
    draw(acc_record, "")
    plt.show()
    print("准确率：", acc_record[-1].item())