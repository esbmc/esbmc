def f():
    return 1


def g():
    return "a"


h = lambda h: h()
assert h(f) == 1
assert h(g) == "b"
