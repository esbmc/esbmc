import _sv_verifier

x = _sv_verifier.nondet_int()
if 1 < x < 4:
    s = sorted([x, 1])
    assert s[0] == 1 and s[1] == x
    r = sorted([x, 1], reverse=True)
    assert r[0] == x and r[1] == 1
