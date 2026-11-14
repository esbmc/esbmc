def dec1(f):
    def wrapper(a, b):
        return f(a, b) + 1
    return wrapper


def dec2(f):
    def wrapper(a, b):
        return f(a, b) + 2
    return wrapper


a = dec1
a = dec2


@a
def func(a, b):
    return a + b


assert func(1, 2) == 5
