def gen():
    yield 1

g = gen()

x = next(g)

try:
    next(g)
    assert False     # Should raise StopIteration
except StopIteration:
    pass
