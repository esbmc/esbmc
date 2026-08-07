def outer(base):
    def inner(*rest):
        return base + len(rest)

    return inner(1, 2)


assert outer(10) == 12
