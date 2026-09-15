class uint64(int):
    pass


x = nondet_int()
__ESBMC_assume(x >= 1 and x <= 64)
p = 2**x
r = uint64(x % p)
assert r > x
