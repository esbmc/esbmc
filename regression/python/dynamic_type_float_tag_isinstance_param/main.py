def scale(v):
    if isinstance(v, int):
        s = 100 // v
        assert s == 25
    return "done"


def run(v):
    return scale(v)


if nondet_bool():
    x = 4
elif nondet_bool():
    x = "a"
else:
    x = 2.5
assert run(x) == "done"
