b: bool = nondet_bool()

l = [42]

if b:
    l.remove(42)
    assert len(l) == 0
else:
    assert len(l) == 1
    assert l[0] == 41
