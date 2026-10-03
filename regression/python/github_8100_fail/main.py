# #8100: appends through a parameter or an alias must reach list ==.
def grow(x: list, v: int) -> None:
    x.append(v)


m: list = [1]
n: list = [1]
grow(m, 1)
grow(n, 2)
assert m == n
