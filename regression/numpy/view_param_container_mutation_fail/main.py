import numpy as np

def collect_views(row, out):
    out.append(row)

a = np.array([1, 2, 3])
v = a[0:]
results = []
collect_views(v, results)
