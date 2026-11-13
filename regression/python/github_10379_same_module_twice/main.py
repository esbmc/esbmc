import pkg.sub
import pkg.sub as other

m = pkg.sub if nondet_bool() else pkg
c = m.func()
if m is other:
    assert c == 1
else:
    assert c == 2.5
