def test_dict():
    d = {1: "2", 2: "4"}
    assert sorted(d.keys()) == [1, 2]
    assert sorted(d.values()) == ['2', '4']
    assert sorted(d.items()) == [(1, '2'), (2, '4')]


def test_items():
    e = {}
    e[4] = 1.0
    assert list(e.items()) == [(4, 1.0)]

    assert sorted(dict(["ab", "cd"]).items()) == [('a', 'b'), ('c', 'd')]
    assert sorted(dict(set([(1, 2.0), (3, 4.0)])).items()) == [(1, 2.0), (3, 4.0)]


def test_dict_fromkeys():
    assert dict.fromkeys([1, 2, 3]) == {1: None, 2: None, 3: None}
    assert dict.fromkeys([1, 2, 3], 7) == {1: 7, 2: 7, 3: 7}
    assert dict.fromkeys([1, 2, 3], 4.0) == {1: 4.0, 2: 4.0, 3: 4.0}
    assert dict.fromkeys([1, 2, 3], "abc") == {1: 'abc', 2: 'abc', 3: 'abc'}


def test_all():
    test_dict()
    test_items()
    test_dict_fromkeys()


if __name__ == "__main__":
    test_all()
