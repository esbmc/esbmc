# A name bound before the call in the branch and a name bound after it are both
# read by the rest of the branch, and a nested if holds the call.
def f():
    return "a"


def g():
    return 1


outer = nondet_bool()
inner = nondet_bool()
h = f if nondet_bool() else g
if outer:
    before = 5
    if inner:
        r = h()
        assert (r == "a") == (h is f)
        after = 1
    else:
        after = 0
    assert before == 5
    assert (after == 1) == inner
else:
    after = 7
assert after != 5
