import numpy as np
from PSO import PSO
from polynomialFunction import PolynomialFunction as F

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    start, stop = -10, 10
    num = 100 * (stop - start)
    
    f = F(w=np.array([10,8,-1,0.1,-0.001]))
    
    x = np.linspace(start, stop, num)
    y = f(x)
    plt.plot(x, y)
    
    ans = PSO(f, start, stop)
    print(ans)
    plt.plot(ans[0], ans[1], "*")
    plt.show()