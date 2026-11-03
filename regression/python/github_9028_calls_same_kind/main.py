def f():
    return "a"


def g():
    return "b"


def run(c):
    if c:
        return f()
    else:
        return g()


assert run(True) == "a"
assert run(False) == "b"
