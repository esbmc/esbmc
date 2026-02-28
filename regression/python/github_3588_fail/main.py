def test_unpacking():
    a, b, c = [1, 2, 3]
    head, *tail = [1]
    assert head == 2 and tail == []


test_unpacking()
