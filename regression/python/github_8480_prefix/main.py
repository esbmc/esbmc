def f(a, b, c=3):
    return a + b + c


def g(*args):
    return f(10, *args)


assert g(1) == 14
assert g(1, 2) == 13
