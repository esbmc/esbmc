from typing import Any

t: Any = (1, 2)
assert t[0] == 1
assert t[1] == 2

d = {1: ('A', 'B'), 2: ('C', 'D')}
v: Any = d[1]
assert v[0] == 'B'

x: Any = (10, 20)
y = x[0]
x = 5
assert x == 5
assert y == 10
