# One imported function returns None and the other 42; each arm of the
# dispatch calls its own.
import pkg.sub as sub
import pkg

c = nondet_bool()
f = sub.func if c else pkg.func
r = f()
if c:
    assert r == 42
else:
    assert r is None
