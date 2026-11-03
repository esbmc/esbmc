def f():
    return "a"


def g():
    return 1


def pick(flag):
    if flag:
        return f
    return g
