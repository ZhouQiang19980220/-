import numpy as np

a = np.arange(12)

a[0:6] = 0
b = a[a<6]
b[0] = 1000000

print("")