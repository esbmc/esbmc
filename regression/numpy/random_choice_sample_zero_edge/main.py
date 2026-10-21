import numpy as np

a = np.random.choice([1, 2], 0, replace=False)

assert len(a) == 0
