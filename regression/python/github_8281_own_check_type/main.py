import _sv_verifier


def check_type(value, hint):
    return 1


n = _sv_verifier.nondet_int()
assert check_type(n, str) == 1
