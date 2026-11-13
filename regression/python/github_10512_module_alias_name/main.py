import helper


def func():
    return "s"


c = True
if c:
    f = helper.alias
else:
    f = func
r = f()
if f.__name__ == "func":
    assert r == 42
