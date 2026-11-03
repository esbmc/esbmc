def helper(x):
    return x * 2


def make():
    g = helper

    def inner(x):
        r = g(x)
        return r

    return inner


h = make()
assert h(4) == 8
