lst = []
n = nondet_int()
__ESBMC_assume(1 <= n <= 3)

i = 0
while i < n:
    lst.append(nondet_int())
    i += 1

v = lst[-1]
assert v == lst[-1]
