a: int = 0
assert len(range(a, 4)) == 4
assert len(range(4, a)) == 0

b: int = 1
assert len(range(b, 10, 3)) == 3
assert len(range(10, b, -3)) == 3
assert len(range(b, 10, -3)) == 0
assert len(range(b, 11, 5)) == 2

n: int = -3
assert len(range(n)) == 0

l: list[int] = list(range(b, 10, 3))
assert len(l) == 3
assert l[0] == 1 and l[1] == 4 and l[2] == 7

r: list[int] = list(range(10, b, -4))
assert r == [10, 6, 2]
