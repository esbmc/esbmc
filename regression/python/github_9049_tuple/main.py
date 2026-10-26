import mymod

for c in [True, False]:
    if c:
        assert mymod.m(1, (int, str))
        assert mymod.m(True, int)
    else:
        assert not mymod.m(1.5, (int, str))
