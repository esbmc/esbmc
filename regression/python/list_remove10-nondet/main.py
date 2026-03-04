b: bool = nondet_bool()

l = [5, 5, 5]

if b:
    l.remove(5)

    # Length reduced by 1
    assert len(l) == 2

    # Remaining elements still equal to 5
    assert l[0] == 5
    assert l[1] == 5
else:
    # No removal
    assert len(l) == 3
    assert l[0] == 5
    assert l[1] == 5
    assert l[2] == 5
