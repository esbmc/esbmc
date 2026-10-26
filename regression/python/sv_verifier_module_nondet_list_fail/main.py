import _sv_verifier

ints = _sv_verifier.nondet_list(_sv_verifier.nondet_int)
copied = [b + 1 for b in ints]
for v in copied:
    assert not isinstance(v, int)
