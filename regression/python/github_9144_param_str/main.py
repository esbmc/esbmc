def run(c, n):
    if c:
        v = n
    else:
        v = "a"
    return v


c = nondet_bool()
if c:
    assert run(c, "b") == "b"
else:
    assert run(c, "b") == "a"
