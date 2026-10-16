def f():
    return "a"


def g():
    return 1


def run(c):
    if c:
        return f()
    else:
        return g()


assert run(False) == "a"
