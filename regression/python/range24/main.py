def test_range_single_arg():
    l = list(range(5))
    assert len(l) == 5
    assert l[0] == 0
    assert l[1] == 1
    assert l[2] == 2
    assert l[3] == 3
    assert l[4] == 4


test_range_single_arg()
