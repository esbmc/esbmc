def f(x):
    return x


def g(*args):
    return f(*args)


g(1, 2)
