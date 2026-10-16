# A list and a tuple of functions returning int and str, called at a nondet
# index and through names unpacked from the tuple.
def f1():
    return 42


def f2():
    return "b"


fs = [f1, f2]
k = nondet_int()
__ESBMC_assume(0 <= k and k < 2)
r = fs[k]()
assert isinstance(r, str) == (k == 1)
x, y = (f2, f1)
s = x()
assert len(s) == 1
assert y() == 42
