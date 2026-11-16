from from_module import *
from math import *
import _sv_verifier

n = _sv_verifier.nondet_int()
_sv_verifier.check_type(n, int)
_sv_verifier.check_type(x, int | None)
_sv_verifier.check_type([n], list[int])
