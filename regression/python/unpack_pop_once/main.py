h = [(0, 5), (1, 2)]
d, n = h.pop()
assert d == 1 and n == 2
assert len(h) == 1

g = [(0, 5)]
a, b = g.pop(0)
assert a == 0 and b == 5
assert len(g) == 0

k = [[3, 4]]
x, y = k.pop()
assert x == 3 and y == 4
