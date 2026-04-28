seed = 2


def f(*, x=seed + 3):
    return x


assert f() == 5
assert f(x=1) == 1
