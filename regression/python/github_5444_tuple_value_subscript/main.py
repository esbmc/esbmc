d = {1: ('A', 'B'), 2: ('C', 'D')}
v = d[1]
assert v[0] == 'A'
assert v[1] == 'B'
assert len(v) == 2

t = ('X', 'Y')
d2 = {1: t}
w = d2[1]
assert w[0] == 'X'

nums = {1: (10, 20)}
p = nums[1]
assert p[0] == 10
assert p[1] == 20
