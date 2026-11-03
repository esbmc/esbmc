import numpy as np

a = np.array([10, 20, 30, 40, 50, 60])
st = nondet_int()
__ESBMC_assume(st == 2)
lo = nondet_int()
__ESBMC_assume(lo == -100)
s = a[lo:5:st]
assert len(s) == 3
assert s[2] == 50
neg = nondet_int()
__ESBMC_assume(neg == -2)
r = a[4:0:neg]
assert len(r) == 2
assert r[0] == 50
assert r[1] == 30
e = a[2:2:st]
assert len(e) == 0
