# github #8090: lists that gain a nested list through a parameter must still
# compare element-wise.


def grow(x: list, v: int) -> None:
    x.append([v])


m: list = ["y"]
n: list = ["y"]
grow(m, 1)
grow(n, 2)
assert m == n
