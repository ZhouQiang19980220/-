import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.01, 50, 0.01)
y = 2 / x + (1 / (x ** 2)) * (np.exp(-2 * x) - 1)

print(y[0])
print(2 / x[0])
plt.plot(x, y)
plt.show()