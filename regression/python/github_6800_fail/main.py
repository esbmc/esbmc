def outer():
    base = 10

    def inner(*rest):
        return base + len(rest)

    return inner(1, 2, 3)


assert outer() == 12
