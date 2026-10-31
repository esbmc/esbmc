for c in [True, False]:
    if c:
        r = "a"
    else:
        r = 1.5
    assert (type(r) is float) == (not c)
