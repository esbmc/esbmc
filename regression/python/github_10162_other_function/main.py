def g():
    max = 64
    return max


def f():
    return max(2, 1)


assert g() == 64
assert f() == 2
