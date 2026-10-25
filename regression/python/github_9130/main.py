# Issue #9130: a bool stored in a name that holds a str on another path
# keeps the bool tag.
for c in [True, False]:
    if c:
        r = True
    else:
        r = "a"
    assert isinstance(r, bool) == c
    assert isinstance(r, int) == c
    assert isinstance(r, str) == (not c)
    if c:
        assert isinstance(r, bool)
        assert r == 1
