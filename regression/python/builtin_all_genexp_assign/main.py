l: list[int] = [1, 2, 3]
b: bool = all(x > 0 for x in l)
assert b

l2: list[int] = [1, -1, 3]
b2: bool = all(x > 0 for x in l2)
assert not b2
