# FAIL: the chain preserves values, so result[0] == v always.
# Asserting result[0] != v must be violated for any value of v.


def gen1(arr):
    for x in arr:
        yield x


def gen2(arr):
    for y in gen1(arr):
        yield y


v = nondet_int()
__ESBMC_assume(v >= 0)
__ESBMC_assume(v <= 1000)

data = [v]
result = list(gen2(data))

assert result[0] != v  # Wrong: chain preserves value, so result[0] == v always
