def test_list_comprehension():
    squares = [x * x for x in range(4)]
    assert squares == [0, 1, 4, 9]

    # Edge case
    assert [x for x in []] == []


test_list_comprehension()
