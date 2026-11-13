# _sv_verifier.nondet_list, called through its module, gives a list of the
# generator's type for each of float, bool and str.
import _sv_verifier

floats = _sv_verifier.nondet_list(_sv_verifier.nondet_float)
for f in floats:
    assert isinstance(f, float)

flags = _sv_verifier.nondet_list(_sv_verifier.nondet_bool)
for g in flags:
    assert isinstance(g, bool)

words = _sv_verifier.nondet_list(_sv_verifier.nondet_str)
for w in words:
    assert isinstance(w, str)
