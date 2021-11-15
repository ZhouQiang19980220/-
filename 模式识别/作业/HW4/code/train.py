import torch
import torch.nn as nn

def train(trainLoader, config, model):
    # 最大迭代次数
    n_epochs = config['n_epochs'] 
    # 优化器
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(),
                                                          lr = config['lr'])
    
    # 损失函数
    citerion = nn.MSELoss()
    
    acc_record = []

    for i in range(n_epochs):
        model.train()
        acc = 0.0
        l = 0
        for datas, targets in trainLoader:
            optimizer.zero_grad()
            preds = model(datas)
            mse_loss = citerion(preds, targets)
            mse_loss.backward()
            optimizer.step()
            _ , preds = torch.max(preds, axis=1)
            _ , targets = torch.max(targets, axis=1)
            acc += (preds == targets).sum()
            l += targets.size()[0]
        acc_record.append(acc / l)
            
    return acc_record