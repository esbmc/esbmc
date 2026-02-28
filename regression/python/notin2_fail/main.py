def test_string_boundaries() -> None:
    s = "start-middle-end"

    # Not in (substring absent)
    assert 'xyz' not in s

    # Not in (exact match absent)
    assert not ('start' in s)
    assert not ('end' in s)

    # Not in (check for 'in' at boundaries)
    assert not ('start' not in s)
    assert not ('end' not in s)


test_string_boundaries()
