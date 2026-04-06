def nondet_int() -> int: ...

# Chained assignment with list comprehension using nondet values
n = nondet_int()
__ESBMC_assume(n >= 0 and n < 4)
a = b = [i + n for i in range(2)]
assert len(a) == 2
assert len(b) == 2
assert a[0] == b[0]
assert a[1] == b[1]
