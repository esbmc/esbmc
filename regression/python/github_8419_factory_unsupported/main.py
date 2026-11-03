def times(n):

    def dec(f):

        def wrapper(x: int) -> int:
            return n * f(x)

        return wrapper

    return dec


@times(3)
def g(x: int) -> int:
    return x


assert g(2) == 6
