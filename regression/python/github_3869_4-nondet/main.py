def f(arr):
    for x in arr:
        for y in f([]):
            yield y
        yield x

a = nondet_int()
assert list(f([a])) == [a]
