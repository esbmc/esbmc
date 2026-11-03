import numpy as np

try:
    np.random.choice(())
except ValueError:
    assert False
