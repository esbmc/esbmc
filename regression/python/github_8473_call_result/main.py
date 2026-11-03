def f():
    return 1


def pick():
    return f


assert pick() is f
