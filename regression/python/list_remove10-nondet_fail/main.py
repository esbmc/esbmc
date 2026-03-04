b: bool = nondet_bool()

l = [5, 5, 5]

if b:
    l.remove(5)
    assert len(l) == 2
    assert l[0] == 5
    assert l[1] == 5
    assert l[2] == 5
else:
    assert len(l) == 3
    assert l[0] == 5
    assert l[1] == 5
    assert l[2] == 5
