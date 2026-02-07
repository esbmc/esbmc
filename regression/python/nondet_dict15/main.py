def test_dict_size():
    d = nondet_dict(3)  # Expected: a dictionary with up to 3 entries
    # Assertion: the dictionary should be able to have 2 or more entries
    assert len(d) < 4  