for c in [True, False]:
    if c:
        r = "a"
    else:
        r = 1
    assert (type(r) == str) == c
    assert (type(r) is int) == (not c)
    assert type(r) != float
