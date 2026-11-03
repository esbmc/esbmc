# One imported function returns None and the other 42, so the call through
# the conditional is refused.
import pkg.sub as sub
import pkg

c = nondet_bool()
f = sub.func if c else pkg.func
r = f()
if not c:
    assert r is None
