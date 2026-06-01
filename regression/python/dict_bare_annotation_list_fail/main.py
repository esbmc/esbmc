def test() -> None:
    # Wrong assertion: the bare-`dict`-annotation literal's value is a list
    # of length 3, but we assert length 99, so verification must fail.
    a: dict = {0: [1, 2, 3]}
    assert len(a[0]) == 99


test()
