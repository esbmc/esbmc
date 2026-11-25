import nested.to_import


def func():
    return 7


for c in [True, False]:
    if c:
        f = nested.to_import.func
    else:
        f = func
    r = f()
    if f is not func:
        assert r == 43
    else:
        assert r == 7
