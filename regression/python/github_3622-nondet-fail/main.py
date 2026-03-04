lst = []
n = nondet_int()
__ESBMC_assume(0 <= n <= 3)

i = 0
while i < n:
    lst.append(nondet_int())
    i += 1

v = nondet_int()
assert lst[-1] == v
