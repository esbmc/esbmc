# The yielded value is only part of the right-hand side, so next() is left
# unmodelled rather than replacing the whole expression with it.
def gen():
    yield 1
    yield 2


g = gen()
x = next(g) + 1
assert x == 2
