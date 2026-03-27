def gen():
    i = 1
    assert i == 1  # local fact
    return  # generator exits immediately
    yield 1  # unreachable
    return  # unreachable


g = gen()

s = 1
assert s == 1

try:
    s = s + next(g)
    assert False  # unreachable: next(g) always raises StopIteration
except:
    # we must enter here
    s = 5
    assert s == 5

res = s

# Final properties
assert res == 5
assert s == 5
