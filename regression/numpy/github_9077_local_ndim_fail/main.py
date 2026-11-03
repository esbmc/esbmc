import numpy as np

data: list[list[float]] = [[0.0, 1.0, 2.0]]
adata = np.array(data)
assert adata.ndim == 1
