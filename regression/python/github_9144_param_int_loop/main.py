def run(c, n):
    if c:
        v = n
    else:
        v = "a"
    return v


for c in [True, False]:
    if c:
        assert run(c, 3) == 3
    else:
        assert run(c, 3) == "a"
