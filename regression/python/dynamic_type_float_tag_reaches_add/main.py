def bump(v):
    if isinstance(v, (int, float)):
        assert v + 1 > 1
    return "done"


def run(v):
    return bump(v)


if nondet_bool():
    x = 4
elif nondet_bool():
    x = "a"
else:
    x = 2.5
assert run(x) == "done"
