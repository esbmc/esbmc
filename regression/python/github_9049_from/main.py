from mymod import m

for c in [True, False]:
    if c:
        assert m(1, int)
    else:
        assert not m(1, str)
