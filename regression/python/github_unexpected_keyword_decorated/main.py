def dec(f):

    def w(**kw):
        return f(1)

    return w


@dec
def inner(v):
    return v


assert inner(x=2) == 1
