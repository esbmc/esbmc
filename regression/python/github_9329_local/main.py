def inner(v):
    return v


def f():
    d = {"v": 3}
    return inner(**d)


assert f() == 3
