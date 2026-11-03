s = []
if nondet_bool():
    s.append(1)
try:
    x = s.pop()
except IndexError:
    x = 0
assert x == 0 or x == 1

t = [1, 2]
i = nondet_int()
try:
    y = t.pop(i)
    assert y == 1 or y == 2
except IndexError:
    assert i >= 2 or i < -2
