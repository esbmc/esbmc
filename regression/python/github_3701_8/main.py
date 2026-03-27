def gen():
    if True:
        return
    yield 1

g = gen()

try:
    next(g)
    assert False
except StopIteration:
    pass
