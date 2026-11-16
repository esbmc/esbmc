s = "s"


def dec(f):
    def wrapper(a, b):
        return f(a, b)

    return wrapper


def raw(a, b):
    return a + b


func = dec(raw)
g = lambda w: s
c = func(g(0), 1)
