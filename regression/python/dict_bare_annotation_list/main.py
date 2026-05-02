def test() -> None:
    # Bare `dict` annotation (no K, V parameters): the value type must be
    # recovered from the literal's first entry so that len(a[k]) routes
    # through list handling instead of defaulting to char*.
    a: dict = {0: [1, 2, 3]}
    assert len(a[0]) == 3
    assert a[0] == [1, 2, 3]


test()
