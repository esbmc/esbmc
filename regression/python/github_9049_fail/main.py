import mymod

for c in [True, False]:
    if c:
        assert mymod.m(1, int)
    else:
        assert mymod.m(1, str)
