d = [[1.5, 2.5], [1.5]]
e = {k for j in d for k in j}
assert len(e) == 2
