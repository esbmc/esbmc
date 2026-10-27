def func(x: int | None = None):
    if x is None:
        return "a"
    return x


r = func()
assert r != "a"
