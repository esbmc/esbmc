base = 40


def f(x=base + 2):
    return x


assert f() == 42
assert f(10) == 10
