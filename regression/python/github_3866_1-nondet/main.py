# Nondet int value flows unchanged through a two-level generator chain.
# Exercises the value-assignment path: yield x -> y = x -> append(y).

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

assert len(result) == 1
assert result[0] == v   # value must be preserved through the chain
assert result[0] >= 0
assert result[0] <= 1000
