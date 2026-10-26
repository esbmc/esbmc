# Issue #9155: int() of a str-or-int name runs only where Python evaluates it:
# the or tail is skipped on the str path, the lambda reads the later value.
for c in [True, False]:
    if c:
        r = "a"
    else:
        r = 1
    ok = c or int(r) == 1
    assert ok
    f = lambda: int(r)
    if not c:
        assert f() == 1
