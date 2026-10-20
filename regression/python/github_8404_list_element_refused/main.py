# The list holds functions returning int and str, called at a nondet index.
def f1():
    return 42


def f2():
    return "b"


fs = [f1, f2]
k = nondet_int()
__ESBMC_assume(0 <= k and k < 2)
r = fs[k]()
assert r == 42 or r == "b"
