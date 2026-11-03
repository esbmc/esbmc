# Issue #9155: int() and float() of a name holding a str or an int
# depending on the path convert the value the path stored.
for c in [True, False]:
    if c:
        r = "12"
    else:
        r = 7
    assert int(r) == (12 if c else 7)
    assert float(r) == (12.0 if c else 7.0)

for d in [True, False]:
    if d:
        s = "x"
    else:
        s = 3
    try:
        assert int(s) == 3
        assert not d
    except ValueError:
        assert d
