def f():
    return "a"


def g():
    return "b"


def pick(flag):
    if flag:
        return f
    return g
