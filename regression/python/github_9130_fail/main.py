# Issue #9130: the stored True is a bool, not an int, so this fails.
for c in [True, False]:
    if c:
        r = True
    else:
        r = "a"
    if c:
        assert not isinstance(r, bool)
