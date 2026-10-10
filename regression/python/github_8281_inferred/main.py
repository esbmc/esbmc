# inc is unannotated and called with an int and a float, so its return type
# is one inference for the whole program; check_type must not let that guess
# raise TypeError on the branch whose value really is the hinted type.
import _sv_verifier


def inc(y):
    return y + 1


if _sv_verifier.nondet_bool():
    a = inc(1)
    _sv_verifier.check_type(a, int)
else:
    a = inc(1.5)
    _sv_verifier.check_type(a, float)
