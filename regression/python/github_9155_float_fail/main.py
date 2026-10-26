# Issue #9155: float() of the str path raises ValueError.
for c in [False, True]:
    if c:
        r = "a"
    else:
        r = 1
    y = float(r)
