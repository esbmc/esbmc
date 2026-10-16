def f(v) -> bool:
    t = str
    return isinstance(v, t)


assert f("a")
assert f(1)
