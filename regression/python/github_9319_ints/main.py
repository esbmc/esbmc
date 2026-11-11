def dec(f):
    def wrapper(a, b):
        return f(a, b)

    return wrapper


@dec
def func(a, b):
    return a + b


c = func(1, 2)
assert c == 3
