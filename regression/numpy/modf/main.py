import numpy as np

frac:float
intg:float
frac, intg = np.modf(3.14)

assert frac + intg >= 3.14
