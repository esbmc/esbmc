# Issue #9155: int() and float() of a call returning a str or an int call it
# once.
calls = 0


def g(e: bool):
    global calls
    calls += 1
    if e:
        return "5"
    return 4


for e in [True, False]:
    assert int(g(e)) == (5 if e else 4)
    assert float(g(e)) == (5.0 if e else 4.0)
assert calls == 4
