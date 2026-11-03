def f():
    return "a"


def g():
    return 1


def run(c):
    if c:
        return f()
    else:
        return g()


x = run(False)
assert isinstance(x, str)
