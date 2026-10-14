import numpy as np

b = np.array([[True, False, True], [False, False, True]])

any0 = np.any(b, axis=0)
any1 = np.any(b, axis=1)
all0 = np.all(b, axis=0)
all1 = np.all(b, axis=1)

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
