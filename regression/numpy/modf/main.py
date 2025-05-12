import numpy as np

frac, intg = np.modf(3.14)
assert np.isclose(frac + intg, 3.14)
