for c in [True, False]:
    if c:
        r = "a"
    else:
        r = 1
    assert type(r) == int
