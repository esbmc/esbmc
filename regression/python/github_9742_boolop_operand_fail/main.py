# GitHub #9742: f([], [1]) is [], so the assertion fails.
def f(x, y):
    return x and y


assert f([], [1]) != []
