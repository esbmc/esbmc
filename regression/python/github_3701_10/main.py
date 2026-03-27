def gen():
    i = 0
    while i < 3:
        yield i
        i += 1


g = gen()

a = next(g)
b = next(g)
c = next(g)

assert a == 0
assert b == 1
assert c == 2
