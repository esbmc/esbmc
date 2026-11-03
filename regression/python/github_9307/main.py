def dec(f):
    def wrapper(a):
        result = f(a)
        return result

    return wrapper


@dec
def func(a):
    return a + 1


c = func(1)
assert c == 2
