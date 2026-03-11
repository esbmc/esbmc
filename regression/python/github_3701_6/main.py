def gen():
    yield 1
    yield 2

g = gen()

x1 = next(g)
x2 = next(g)

assert x1 == 1
assert x2 == 2
