for c in [True, False]:
    if c:
        r = "a"
    else:
        r = 1
    if c:
        assert type(r) == str
    else:
        assert type(r) == int
