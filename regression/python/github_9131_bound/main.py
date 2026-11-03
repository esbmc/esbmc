for c in [True, False]:
    if c:
        r = "a"
    else:
        r = 1
    t = type(r)
    assert (t == str) == c
    assert (t is int) == (not c)
    assert (t != str) == (not c)
