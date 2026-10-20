def twice(f):

    def wrapper(x):
        return f(f(x))

    return wrapper


@twice
def add1(f):

    def wrapper(x: int) -> int:
        return f(x) + 1

    return wrapper


@add1
def g(x: int) -> int:
    return x


assert g(1) == 3
