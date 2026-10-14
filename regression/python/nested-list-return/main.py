# A nested list returned from a function must keep correct element values and
# element types (int/float) when subscripted in the caller (#5103, #5102).
# Previously the frontend referenced the callee-frame inner-list symbol, which
# is nondet in the caller's frame: floats hit a __ESBMC_float_buf OOB crash and
# ints read a wrong value.


def build_float():
    return [[1.5, 2.5]]


def build_int():
    return [[7, 8]]


def build_deep():
    return [[[9.0]]]


Qf = build_float()
Qi = build_int()
Qd = build_deep()

assert Qf[0][0] == 1.5
assert Qf[0][1] == 2.5
assert Qi[0][0] == 7
assert Qi[0][1] == 8
assert Qd[0][0][0] == 9.0

# Variable (non-constant) outer index into a returned nested float list must
# also read the inner element type/value at runtime.
k = 0
assert Qf[k][1] == 2.5
