w = "abcdef"
k = nondet_int()
__ESBMC_assume(k == 0)
caught = 0
try:
    r = w[::k]
except ValueError:
    caught = 1
assert caught == 0
