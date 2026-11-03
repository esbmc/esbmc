def dec(f):
    def wrapper(a, b):
        return f(a=a, b=b)

    return wrapper


def raw(a, b):
    return a + b


func = dec(raw)
c = func("a", 1)
