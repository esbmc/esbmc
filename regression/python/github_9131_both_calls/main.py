for c in [True, False]:
    if c:
        r = "a"
    else:
        r = 1
    assert (type(r) == type("x")) == c
    assert (type(r) == type(1)) == (not c)
