l: list[int] = [1, -1, 3]
b: bool = all(x > 0 for x in l)
assert b
