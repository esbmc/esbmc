import _sv_verifier

x = _sv_verifier.nondet_int()
if 0 < x < 3:
    s = sorted([x, 2, x, 0])
    assert s[0] == 0 and s[1] == x and s[2] == x and s[3] == 2
    assert len(s) == 4
