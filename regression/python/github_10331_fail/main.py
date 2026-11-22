# #10331: the call is evaluated, so b is "neg" and not 0.
import _sv_verifier


def func(x):
    if x > 0:
        return x
    else:
        return "neg"


b = func(-abs(_sv_verifier.nondet_int()))
assert b == 0
