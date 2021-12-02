import numpy as np

# 多项式函数
class PolynomialFunction():
    
    # w是各阶系数，[start,stop]是定义域
    def __init__(self, w, start=-1, stop=1):
        assert len(w.shape) == 1
        self.w = w
        self.start = start
        self.stop = stop
        self.n = self.w.shape[0]
        
    def __call__(self, x):
        if not hasattr(x, '__iter__'):
            return self._f(x)
        else:
            ans = []
            for i in x:
                ans.append(self._f(i))
        return np.array(ans)
    
    def _f(self, x):
        assert self.start <= x <= self.stop
        temp = [x ** i for i in range(self.n)]
        return np.dot(np.array(temp), self.w)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    w = np.array([3, -2, -5, 1])
    start, stop = -2, 5
    f = PolynomialFunction(w, start, stop)
    
    x = np.linspace(start, stop, num=100) 
    y = f(x)
    
    plt.plot(x, y)
    plt.show()
    
    