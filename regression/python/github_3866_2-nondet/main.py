# nondet_bool conditional inside the outer generator while iterating over gen1.
# Exercises the branch inside the inlined body after yield replacement.

def gen1(arr):
    for x in arr:
        yield x

def gen2(arr):
    for y in gen1(arr):
        if nondet_bool():
            yield y
        else:
            yield y + 1

v = nondet_int()
__ESBMC_assume(v >= 0)
__ESBMC_assume(v <= 500)

data = [v]
result = list(gen2(data))

assert len(result) == 1
# gen2 yields either v or v+1 depending on the nondet branch
assert result[0] == v or result[0] == v + 1
assert result[0] >= 0
assert result[0] <= 501
