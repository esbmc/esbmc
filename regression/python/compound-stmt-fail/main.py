def test_basic_while_counting():
    count = 0
    while count < 5:
        count += 1
    assert count == 5, "Loop did not increment count to 5"
    assert 5 / count > 2


test_basic_while_counting()
