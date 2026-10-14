# The part of #6256 that already works: a nested function that escapes but
# captures nothing, and one that captures but is called before escaping.
# Pinning these keeps the closure fix honest -- it must not regress what the
# frontend already models.


def outer():
    def inner(x):
        return x + 1

    return inner


f = outer()
assert f(3) == 4


def applied(n):
    def add(x):
        return x + n

    return add(3)


assert applied(5) == 8
