def test_empty_dict():
    d = {}
    e = {}

    assert not d
    assert not e
    assert len(d) == len(e) == 0

test_empty_dict()
