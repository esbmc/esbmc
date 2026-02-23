def test_range_uneven_step():
    l = list(range(0, 10, 3))
    assert len(l) == 4
    assert l[0] == 0
    assert l[1] == 2
    assert l[2] == 6
    assert l[3] == 9

test_range_uneven_step()
