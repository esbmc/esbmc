m: list = [0]
n: list = [0]
m[0] = (0,)
assert m != n
assert m[0] == (0,)
m[0] = 5
assert m == [5]

p: list = [(1, 2), (3, 4)]
p[1] = 7
assert p[1] == 7
assert p == [(1, 2), 7]
