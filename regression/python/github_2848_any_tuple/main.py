from typing import Any

t: Any = (1, 2)
assert t[0] == 1
assert t[1] == 2

d = {1: ('A', 'B'), 2: ('C', 'D')}
v: Any = d[1]
assert v[0] == 'A'

x: Any = (10, 20)
y = x[0]
x = 5
assert x == 5
assert y == 10


def f() -> None:
    w: Any = (3, 4)
    assert w[1] == 4


f()

# Re-annotation keeps prior uses intact (the re-bound tuple itself still
# reads through void* — pre-existing limitation, needs the fresh-alias
# mechanism).
a: Any = 5
b = a
a: Any = (7, 8)
assert b == 5
