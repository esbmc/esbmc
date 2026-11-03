# Issue #9155: int() truncates the float path's 2.75 to 2, not 3.
for c in [True, False]:
    if c:
        r = "12"
    else:
        r = 2.75
    assert int(r) == (12 if c else 3)
