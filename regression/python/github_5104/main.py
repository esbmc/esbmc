# GitHub #5104: a user function returning a float (via a ternary) whose
# argument is itself another user function's return value crashed the SMT
# backend in an FP comparison. The crash came from the parameter and return
# being typed as a pointer because b()'s return type could not be inferred:
# its `return v` local was resolved in the wrong scope.
def myabs(x):
    return x if x >= 0.0 else -x


def b():
    v = nondet_float()
    __ESBMC_assume(v >= -10.0)
    __ESBMC_assume(v <= 10.0)
    return v


a = b()
assert myabs(a) <= 100.0
