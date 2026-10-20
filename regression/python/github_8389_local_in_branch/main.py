def f(v, c: bool) -> bool:
    if c:
        t = int
    else:
        t = str
    return isinstance(v, t)


assert f(1, True)
assert f("a", False)
