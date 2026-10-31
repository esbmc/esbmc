def f(xs):
    return xs


assert f((1, "b", 2.5))[1] == "b"
assert f((1, "b", 2.5))[0] == 1
