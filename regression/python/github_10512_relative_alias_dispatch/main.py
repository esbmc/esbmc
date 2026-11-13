import nested.to_import


def func():
    return "s"


c = True
if c:
    f = nested.to_import.func
else:
    f = func
r = f()
if f is func:
    assert r == "s"
else:
    assert r == 42
