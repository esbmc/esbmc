import math

l:list[float] = nondet_list(2, nondet_float())
__ESBMC_assume(len(l)>0)
x = l[0]
if not math.isnan(x):
    l.append(x)

assert len(l) > 0
assert l[0] == 2.2
