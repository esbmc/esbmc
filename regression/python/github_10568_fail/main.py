def g():
    return 1


def func(flag):
    if flag:
        return g
    return None


a = func(False)
a()
