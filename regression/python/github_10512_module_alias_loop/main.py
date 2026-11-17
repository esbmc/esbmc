import helper


def func():
    return "s"


for c in [True, False]:
    if c:
        f = helper.alias
    else:
        f = func
    r = f()
    if f is not func:
        assert r == 42
