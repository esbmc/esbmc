# h holds what dec returns, which takes y, so h(x=1) raises TypeError.
def dec(fn):

    def w(y):
        return fn(y)

    return w


@dec
def f(x):
    return x


h = f
assert h(x=1) == 1
