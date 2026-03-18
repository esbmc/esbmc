# Recursive generator with nondet elements — should pass
def flatten(arr):
    for x in arr:
        for y in flatten([]):
            yield y
        yield x

x: int
y: int
z: int
assert list(flatten([x, y, z])) == [x, y, z]
