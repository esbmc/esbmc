# Recursive generator with a single nondet int element.
# Exercises the simplest list_eq path: one element, one boundary check iteration.
def f(arr):
    for x in arr:
        for y in f([]):
            yield y
        yield x

a: int
assert list(f([a])) == [a]
