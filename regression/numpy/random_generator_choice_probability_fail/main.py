import numpy as np

rng = np.random.default_rng(4)
rng.choice([1, 10], p=[-0.1, 1.1])
