d = {1: ('A', 'B'), 2: ('C', 'D')}
v = d.get(1)
assert v is not None
assert v[0] == 'A'
assert v[1] == 'B'

m = d.get(3)
assert m is None

nums = {1: (10, 20)}
w = nums.get(2, (0, 0))
assert w[0] == 0
u = nums.get(1, (0, 0))
assert u[1] == 20
