d = [[1.5], [2.5]]
assert [1.5] in d
assert [3.5] not in d
e = {k for j in d for k in j}
assert len(e) == 2
