import numpy as np

def read_and_return(row):
    x = row[0]
    return x + 1

a = np.array([5, 10])
v = a[0:]
result = read_and_return(v)
assert result == 5  # wrong: result is 6
