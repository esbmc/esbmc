def test_concat():
    s = nondet_str()
    result = s + "suffix"
    assert len(result) >= 6  # At least the suffix length


test_concat()
