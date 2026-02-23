n: int = 3
a: list[int] = [1, 3, 2]

m = max(a)

assert m >= a[0]
assert m >= a[1]
assert m >= a[2]
