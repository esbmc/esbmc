import numpy as np


def stride_access_2d(a):
    # This requires proper 2-D stride handling in parameters
    # Access using negative indices on 2-D array
    last_row = a[-1]  # Should be the last row
    last_elem = a[-1, -1]  # Should be the last element
    return last_row[0] + last_elem


a = np.array([[1, 2], [3, 4]])
result = stride_access_2d(a)

# a[-1] is [3, 4], so a[-1][0] = 3
# a[-1, -1] = 4
# result = 3 + 4 = 7
assert result == 7
