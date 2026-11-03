for c in [True, False]:
    if c:
        r = "a"
    else:
        r = 1
    t = type(r)
    assert (t == str) == (not c)
