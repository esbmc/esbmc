# GitHub #5104 (negative): with the parameter/return now correctly typed as
# float, a genuinely violated FP bound must still be reported FAILED — the fix
# must not mask real violations. |a| can reach 10.0, which exceeds 5.0.
def myabs(x):
    return x if x >= 0.0 else -x


def b():
    v = nondet_float()
    __ESBMC_assume(v >= -10.0)
    __ESBMC_assume(v <= 10.0)
    return v


a = b()
assert myabs(a) <= 5.0
