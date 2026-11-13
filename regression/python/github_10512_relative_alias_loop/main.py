import nested.to_import


def func():
    return "s"


for c in [True, False]:
    if c:
        f = nested.to_import.func
    else:
        f = func
    r = f()
    if f is not func:
        assert r == 42
