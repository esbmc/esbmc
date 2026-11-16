import numpy as np

rng = np.random.default_rng(1)
rng.integers(0, 128, dtype=np.int8, endpoint=True)
