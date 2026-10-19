# Values from the module-qualified nondet functions type op's calls as
# the unqualified forms do, so op is specialized for int and float.
import _sv_verifier


def op(a, b):
    return a + b


r1 = op(_sv_verifier.nondet_int(), 1)
r2 = op(_sv_verifier.nondet_float(), 1)
assert isinstance(r1, int)
assert isinstance(r2, int)
