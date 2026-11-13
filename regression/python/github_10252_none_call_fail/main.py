def g():
    return 1


def func(flag):
    if flag:
        return g


a = func(False)
a()