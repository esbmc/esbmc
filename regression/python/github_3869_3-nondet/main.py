def f(arr):
    for x in arr:
        for y in f([]):
            yield y
        yield x

a = nondet_float()
assert list(f([a])) == [a]
