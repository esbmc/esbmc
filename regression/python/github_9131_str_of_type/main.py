for c in [True, False]:
    if c:
        r = "a"
    else:
        r = 1
    assert (str(type(r)) == "<class 'str'>") == c
