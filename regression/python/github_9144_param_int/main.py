def run(c, n):
    if c:
        v = n
    else:
        v = "a"
    return v


c = nondet_bool()
if c:
    assert run(c, 3) == 3
else:
    assert run(c, 3) == "a"
