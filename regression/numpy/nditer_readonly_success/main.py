import numpy as np

total = 0
for x in np.nditer(np.array([[1, 2], [3, 4]])):
    total = total + x

assert total == 10
