import matplotlib.pyplot as plt
import numpy as np

# 绘制10s内样本函数，频率是100
f = 100
time = 10

lam = 1
# 时间间隔服从指数分布：这里画5个点
num = 10
np.random.seed(0)
r = np.random.exponential(1 / lam, size=num)
t = np.hstack([[0], np.cumsum(r)])
print(t)

j = 0
x = np.linspace(0,time,f*time)
y = np.cos(x)
for i in range(num-1):
    start = round(t[i]*f)
    stop = round(t[i+1]*f)
    if start >= 0 and stop <= time * f:
        plt.plot(x[start:stop],y[start:stop],'--')
        y *= -1
        plt.plot(x[start:stop],y[start:stop])
plt.show()