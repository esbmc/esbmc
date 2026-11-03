import numpy as np


class Kinds:
    int32 = "not a dtype"


a = np.zeros(2, dtype=Kinds.int32)
