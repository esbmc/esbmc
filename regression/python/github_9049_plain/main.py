import mymod

for c in [True, False]:
    if c:
        assert mymod.m(1)
    else:
        assert not mymod.m("a")
