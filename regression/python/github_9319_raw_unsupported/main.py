def dec(f):
    def wrapper(a, b):
        return f(a, b)

    return wrapper


def raw(a, b):
    return a + b


func = dec(raw)
c = func("a", "b")
assert len(c) == 2
