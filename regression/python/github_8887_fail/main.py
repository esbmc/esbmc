import helper

c = helper.nondet_bool()
if c:
    x = 1
else:
    x = helper.mk()
if not c:
    assert not isinstance(x, str)
