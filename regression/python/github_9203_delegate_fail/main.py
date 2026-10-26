def inner():
    yield 1
    yield 2


def outer():
    yield from inner()


total = 0
for x in outer():
    total += x
assert total == 4
