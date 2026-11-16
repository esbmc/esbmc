d = [[1.5, 2.5], [3.5]]
e = {k for j in d for k in j}
assert len(e) == 3
assert 2.5 in e
