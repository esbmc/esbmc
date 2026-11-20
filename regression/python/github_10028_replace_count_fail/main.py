n: int = nondet_int()
__ESBMC_assume(n >= -2 and n <= 3)
s = "xxx".replace("x", "y", n)
assert s == "yyy"
