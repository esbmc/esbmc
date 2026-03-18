# Two independent nondet_int values in the source list flow through the chain.
# Exercises multiple loop iterations and verifies element order is preserved.

def gen1(arr):
    for x in arr:
        yield x

def gen2(arr):
    for y in gen1(arr):
        yield y

a = nondet_int()
b = nondet_int()
__ESBMC_assume(a >= 0)
__ESBMC_assume(a <= 100)
__ESBMC_assume(b >= 0)
__ESBMC_assume(b <= 100)
__ESBMC_assume(a != b)   # ensure distinct values to verify order

data = [a, b]
result = list(gen2(data))

assert len(result) == 2
assert result[0] == a   # first element preserved
assert result[1] == b   # second element preserved, order maintained
