import mymod

f = mymod.m
for c in [True, False]:
    if c:
        assert f(1, int)
    else:
        assert not f(1, str)
