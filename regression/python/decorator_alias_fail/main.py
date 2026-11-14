def dec1(f):
    def wrapper(a, b):
        return f(a, b) + 1
    return wrapper


a = dec1


@a
def func(a, b):
    return a + b


assert func(1, 2) == 3
