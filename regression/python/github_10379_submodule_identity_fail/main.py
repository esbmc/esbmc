import pkg.sub

m = pkg.sub if nondet_bool() else pkg
c = m.func()
if m is pkg:
    assert c == 1
