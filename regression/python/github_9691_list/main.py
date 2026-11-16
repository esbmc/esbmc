def g(x: list[int]):
    return 3 * x


def h(v):
    return v + v


r = g([2, 3])
w = h(r[0])
assert isinstance(w, int)
assert w == 4
