def g():
    return 1


def func(flag):
    if flag:
        return g
    return 42


a = func(False)
a()
