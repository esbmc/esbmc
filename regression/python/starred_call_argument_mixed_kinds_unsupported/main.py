# The parameter is passed a list at one call and a tuple at the other, so it
# has no single type to index.
def add(a, b):
    return a + b


def check(args, expected):
    assert add(*args) == expected


check([1, 2], 3)
check((5, 5), 10)
