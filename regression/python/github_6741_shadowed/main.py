def outer():
    def inner(*rest):
        return len(rest)

    return inner(1, 2, 3)


def inner(a, b):
    return a + b


assert outer() == 3
assert inner(1, 2) == 3
