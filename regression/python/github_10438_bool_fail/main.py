def h(x):
    return x + x


def g(a):
    return h(a)


assert g(1) == 2
assert g(1.5) == 3.0
assert g(True) != 2
