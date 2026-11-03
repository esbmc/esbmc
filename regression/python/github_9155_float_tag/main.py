# Issue #9155: int() and float() of a name holding a str or a float
# depending on the path.
for c in [True, False]:
    if c:
        r = "12"
    else:
        r = 2.75
    assert int(r) == (12 if c else 2)
    assert float(r) == (12.0 if c else 2.75)
