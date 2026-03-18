# Recursive generator with nondet elements — should pass
def flatten(arr):
    for x in arr:
        for y in flatten([]):
            yield y
        yield x

x = nondet_int()
y = nondet_int()
z = nondet_int()
assert list(flatten([x, y, z])) == [x, y, z]
