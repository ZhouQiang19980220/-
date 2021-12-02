import numpy as np
from numpy.core.fromnumeric import shape

# 粒子群算法求函数最大值
def PSO(f, start=-1, stop=1):
    
    # 粒子个数
    num = round(stop - start) * 100
    
    # 初始化位置和速度
    x = np.random.rand(num)
    x = (x * (stop - start)) + start
    v = np.random.rand(num)
    
    # 初始化历史最佳和全局最佳
    best_his = f(x)
    best_his_x = x.copy()
    
    best_index = np.argmax(best_his)
    best_glo = best_his[best_index]
    best_glo_x = best_his_x[best_index]  
    
    # 最大迭代次数
    max_epoch = 10000
    for epoch in range(max_epoch):
        
        # 更新位置和速度
        x = x + v
        # 防止越界
        x[x < start] = start
        x[x > stop] = stop
        c = [2, 2]
        r = np.random.rand(2)
        
        v = v + c[0] * r[0] * (best_his_x - x) + c[1] * r[1] * (best_glo_x - x)
        
        y = f(x)
        
        # 更新历史最优
        needRenew = y > best_his
        best_his[needRenew] = y[needRenew]
        best_his_x[needRenew] = x[needRenew]
        
        best_index = np.argmax(best_his)

        # 更新全局最优
        if best_his[best_index] > best_glo:
            best_glo = best_his[best_index]
            best_glo_x = best_his_x[best_index]
            
    return best_glo_x, best_glo
    