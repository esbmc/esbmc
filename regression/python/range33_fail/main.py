def test_range_step_minus_one():
    l = list(range(5, 0, -1))
    assert len(l) == 5
    assert l[0] == 5
    assert l[4] == 0

test_range_step_minus_one()
