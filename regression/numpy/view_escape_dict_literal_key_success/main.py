import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
box = {'row': a[0], 'count': 2}
assert box['row'][2] == 3
assert box['count'] == 2
