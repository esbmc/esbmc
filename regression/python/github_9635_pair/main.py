d = [[1.1, 2.5], [1.2, 2.0]]
e = {(i, i * 2) for j in d for i in j}
assert len(e) == 4
