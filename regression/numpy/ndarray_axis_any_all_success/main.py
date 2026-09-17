import numpy as np

b = np.array([[True, False, True], [False, False, True]])

any0 = b.any(axis=0)
any1 = b.any(axis=1)
all0 = b.all(axis=0)
all1 = b.all(axis=1)

assert any0[0] == True
assert any0[1] == False
assert any0[2] == True
assert any1[0] == True
assert any1[1] == True
assert all0[0] == False
assert all0[1] == False
assert all0[2] == True
assert all1[0] == False
assert all1[1] == False
