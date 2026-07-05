v1: int = nondet_int()
v2: int = nondet_int()
__ESBMC_assume(1 <= v1 <= 100)
__ESBMC_assume(1 <= v2 <= 100)

g = {('A', 'B'): v1, ('C', 'D'): v2}
n = 0
for k, w in g.items():
    n += 1
assert n == 2
