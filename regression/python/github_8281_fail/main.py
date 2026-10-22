import _sv_verifier
from _sv_verifier import check_type

n = _sv_verifier.nondet_int()
check_type([n, "x"], list[int])
