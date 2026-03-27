def test_reverse_preserves_values():
    x = nondet_bool()
    y = nondet_bool()
    z = nondet_bool()
    l = [x, y, z]
    l.reverse()
    assert l[0] == z
    assert l[1] == y
    assert l[2] == x

test_reverse_preserves_values()

