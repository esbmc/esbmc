d = [[1, 2], [2, 3]]
e = {k for j in d for k in j}
assert len(e) == 3
assert 3 in e
