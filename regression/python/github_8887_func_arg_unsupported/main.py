import helper


def identity(v):
    return v


c = helper.nondet_bool()
if c:
    x = 1
else:
    x = helper.mk()
y = identity(x)
if not c:
    assert y == "a"
