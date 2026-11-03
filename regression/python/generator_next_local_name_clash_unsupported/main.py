# The generator's local n is not the module's n, so its body cannot be
# inlined at module level.
def gen():
    n = 1
    while True:
        yield n
        n += 1


n = 10
g = gen()
x = next(g)
assert x == 1
assert n == 10
