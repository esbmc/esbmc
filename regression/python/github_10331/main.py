# #10331: abs() in an argument of a call to a function with mixed return types.
import _sv_verifier


def func(x):
    if x > 0:
        return x
    else:
        return "neg"


b = func(-abs(_sv_verifier.nondet_int()))
assert b == "neg"
