def g(x, k):
    if k == 0:
        return x
    return g(x, k - 1)


def h(y):
    return g(y, 2)


assert g("a", 1) == "a"
assert h(3) == 3
