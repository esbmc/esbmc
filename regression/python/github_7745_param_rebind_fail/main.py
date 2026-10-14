# An enclosing parameter binds the name too, so its type is not inferred (#7745).
def f(x):
    g = lambda n: n + 0
    y = g(x)
    x = 1
    return y

assert f(2.5) == 2
