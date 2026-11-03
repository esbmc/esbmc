import _sv_verifier
import other

n = _sv_verifier.nondet_int()
assert other.check_type(n, str) == 42
