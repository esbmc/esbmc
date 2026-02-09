def test_range_three_args_positive_step():
    l = list(range(0, 10, 2))
    assert len(l) == 5
    assert l[0] == 2
    assert l[2] == 4
    assert l[4] == 8

test_range_three_args_positive_step()
