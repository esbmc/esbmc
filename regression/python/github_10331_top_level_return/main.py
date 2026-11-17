# #10331: the same call when the str return is at top level.
import _sv_verifier


def func(x):
    if x > 0:
        return x
    return "neg"


b = func(-abs(_sv_verifier.nondet_int()))
assert b == "neg"
