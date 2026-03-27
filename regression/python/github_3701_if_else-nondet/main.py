flag = nondet_bool()


def gen():
    if flag:
        yield 1
    else:
        yield 2


g = gen()

x = next(g)

# Should match real Python semantics
assert x == 1 or x == 2

try:
    next(g)
    assert False  # must raise StopIteration
except StopIteration:
    pass
