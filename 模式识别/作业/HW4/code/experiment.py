import os
from torch.utils.data import DataLoader
from config import config
from MyDataSet import MyDataSet
from dnn import Net
from train import train
from utils import draw
import matplotlib.pyplot as plt

# 加载数据集
data_path = os.path.join("data", "data.txt")
target_path = os.path.join("data", "target.txt")
trainSet = MyDataSet(data_path, target_path)
trainLoader = DataLoader(trainSet, batch_size = config['batch_size'])

def ex_batch_size():
    batch_size = [1,2,4,8,16]
    for b_s in batch_size:
        config['batch_size'] = b_s
        # 训练并测试
        net = Net(3, config['hidden_channels'], 3)
        acc_record = train(trainLoader, config, net)
        hint = "batch_size = " + str(b_s)
        draw(acc_record, hint)
    plt.legend()
    plt.show()
    
def ex_learn_rate():
    learn_rate = [1,0.1,0.01,0.001,0.0001]
    for lr in learn_rate:
        config['lr'] = lr
        # 训练并测试
        net = Net(3, config['hidden_channels'], 3)
        acc_record = train(trainLoader, config, net)
        hint = "learn_rate = " + str(lr)
        draw(acc_record, hint)
    plt.legend()
    plt.show()
    
def ex_hidden():
    hidden_channels = [5,10,15,20,25]
    for hc in hidden_channels:
        config['hidden_channels'] = hc
        # 训练并测试
        net = Net(3, config['hidden_channels'], 3)
        acc_record = train(trainLoader, config, net)
        hint = "hidden_channels = " + str(hc)
        draw(acc_record, hint)
    plt.legend()
    plt.show()

        
if __name__ == '__main__':
    ex_batch_size()
    # ex_learn_rate()
    # ex_hidden()