def dec(f):
    def wrapper(a, b):
        result = f(b, a)
        return result

    return wrapper


@dec
def func(a, b):
    return a - b


c = func(5, 3)
assert c == -2
