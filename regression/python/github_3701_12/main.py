def gen():
    i = 0
    while i < 3:
        if i % 2 == 0:
            yield i
        else:
            yield i + 10
        i = i + 1

g = gen()

a = next(g)
b = next(g)
c = next(g)

# Real Python results:
# i=0 → yield 0
# i=1 → yield 11
# i=2 → yield 2

assert a == 0
assert b == 11
assert c == 2

try:
    next(g)
    assert False
except StopIteration:
    pass
