import mymod

for c in [True, False]:
    if c:
        assert mymod.m(1, int)
        assert mymod.m("a", str)
    else:
        assert not mymod.m(1, str)
        assert not mymod.m("a", int)
