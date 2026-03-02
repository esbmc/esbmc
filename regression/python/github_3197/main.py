def f(x):
    if x == 0:
        return 0
    return f(x - 1)


assert f(3) == 0
