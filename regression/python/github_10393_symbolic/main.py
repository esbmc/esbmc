import _sv_verifier

x = _sv_verifier.nondet_int()
if 0 < x < 3:
    s = sorted([x, 1.5, 0.5])
    assert s[0] == 0.5
