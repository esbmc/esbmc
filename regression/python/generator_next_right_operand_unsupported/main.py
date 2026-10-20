# next() as the right operand of arithmetic is left unmodelled, as it is
# on the left.
def gen():
    yield 1
    yield 2


g = gen()
x = 1 + next(g)
assert x == 2
