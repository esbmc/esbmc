import ma
import mb

b = nondet_bool()
m = ma if b else mb
c = m.func()
if b:
    assert isinstance(c, int)
else:
    assert isinstance(c, float)
