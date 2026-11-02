condition = True
q = [3] if condition else [2]
assert isinstance(q[0], int)
assert q == [3]
