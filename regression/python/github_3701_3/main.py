def gen():
    i = 1
    assert i == 1  # local fact
    return  # generator exits immediately
    yield 1  # unreachable
    return  # unreachable


g = gen()

# g is a generator object
assert g is not None
