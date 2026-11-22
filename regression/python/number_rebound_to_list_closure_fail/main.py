def outer() -> bool:
    n = 5

    def inner() -> bool:
        return isinstance(n, int)

    n = [10]
    return inner()


assert outer()
